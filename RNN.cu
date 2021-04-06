#include "RNN.h"
#include "kernels.h"
#include "fp16_emu.h"
#include "fp16_extras.h"

#include <cuda.h>
#include <curand.h>

#include "cub/cub.cuh" 

#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <stdexcept>


template class cudaRNN::RNN<int, half1>;
template class cudaRNN::RNN<int, float>;
template class cudaRNN::RNN<int, double>;
template class cudaRNN::RNN<float, half1>;
template class cudaRNN::RNN<float, float>;
template class cudaRNN::RNN<float, double>;
template class cudaRNN::RNN<double, half1>;
template class cudaRNN::RNN<double, float>;
template class cudaRNN::RNN<double, double>;


namespace cudaRNN
{
    template <typename T_NN> __inline__ cudnnDataType_t getcuDNNDataType();
    template <> __inline__ cudnnDataType_t getcuDNNDataType<double>() { return CUDNN_DATA_DOUBLE; }
    template <> __inline__ cudnnDataType_t getcuDNNDataType<float>()  { return CUDNN_DATA_FLOAT;  }
    template <> __inline__ cudnnDataType_t getcuDNNDataType<half1>()  { return CUDNN_DATA_HALF;   }

    template <typename T_NN> __inline__ cudaDataType_t getcuBLASDataType();
    template <> __inline__ cudaDataType_t getcuBLASDataType<double>() { return CUDA_R_64F; }
    template <> __inline__ cudaDataType_t getcuBLASDataType<float>()  { return CUDA_R_32F; }
    template <> __inline__ cudaDataType_t getcuBLASDataType<half1>()  { return CUDA_R_16F; }

    namespace details
    {
        void checkGPUAlloc(cudaError_t stat) {
            if(stat != CUDA_SUCCESS) throw std::bad_alloc();
        }

        void float2halfArr(float* fArr, half1* halfArr, int nElems)
        {
            dim3 nThreads(256);
            dim3 nBlocks( (nElems + nThreads.x - 1) / nThreads.x );
        
            float2halfArr_ker<<<nBlocks, nThreads>>>(fArr, halfArr, nElems);
        }
    } // namespace details
} // namespace cudaRNN

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::checkOptionErr()
{
    cudnnDataType_t dataType = getcuDNNDataType<T_NN>();

    if((dataType == CUDNN_DATA_HALF   && (options.mathPrecision != MathPrecision::Half && options.mathPrecision != MathPrecision::Float)) ||
       (dataType == CUDNN_DATA_FLOAT  && (options.mathPrecision != MathPrecision::Float)) ||
       (dataType == CUDNN_DATA_DOUBLE && (options.mathPrecision != MathPrecision::Double))) {
        throw std::invalid_argument("Incompatible data type and math precision.");
    }

    if((dataType == CUDNN_DATA_FLOAT  && (options.mathType != MathType::Default && options.mathType != MathType::TensorcoreConv)) ||
       (dataType == CUDNN_DATA_DOUBLE && (options.mathType != MathType::Default))) {
        throw std::invalid_argument("Incompatible data type and math type.");
    }

    if(options.algorithm != Algorithm::Standard && options.biasMode != BiasMode::Double) {
        throw std::invalid_argument("Only double bias mode is supported with a persistent algorithm.");
    }

    if(options.algorithm != Algorithm::Standard && dataType != CUDNN_DATA_FLOAT) {
        throw std::invalid_argument("Only float data type is supported with a persistent algorithm.");
    }

    if(options.inputMode == InputMode::Skip && options.inVecSize != options.hiddenSize) {
        throw std::invalid_argument("Input vector size does not match input hidden size.");
    }

    if(!options.seqLength || !options.inLength || !options.outLength || !options.numLayers || !options.inVecSize || !options.outVecSize ||
       !options.hiddenSize || !options.miniBatchSz) {
        throw std::invalid_argument("Length or size cannot be zero.");
       }

    if(options.inLength > options.seqLength || options.outLength > options.seqLength) {
        throw std::invalid_argument("Input or output sequences cannot exceed sequence length.");
    }
    
    if(0.0f > options.dropout || options.dropout > 1.0f) {
        throw std::invalid_argument("Dropout probability out of bounds.");
    }
}

template <typename T_DATA, typename T_NN> cudaRNN::RNN<T_DATA, T_NN>::RNN(RNNOptions_t _options) : options(_options)
{   
    try {
        checkOptionErr();
        setSpOptions();    

        cudnnCreate(&cudnnHandle);
        cublasCreate(&cublasHandle);

        dimHidden[0] = options.numLayers;
        dimHidden[1] = options.miniBatchSz;
        dimHidden[2] = options.hiddenSize;
        strideHidden[0] = dimHidden[1] * dimHidden[2];
        strideHidden[1] = dimHidden[2];
        strideHidden[2] = 1;

        setDescriptors();

        cudnnGetRNNWeightSpaceSize(cudnnHandle, descriptors.rnn, &wgtSpaceByteSize);
        RNNWgtTensorSz = wgtSpaceByteSize/sizeof(T_NN);

        allocTensors();
        
        // setup curand context
        curandGenerator_t rng;
        curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(rng, wgtSeed);

        // init random weights and biases
        double stddev = sqrt(2. / (options.inVecSize + options.hiddenSize)); // glorot_uniform like tensorflow    

        if(std::is_same<T_NN, half1>::value)  
        {
            float* temp_w;
            cudaMalloc(&temp_w, RNNWgtTensorSz * sizeof(float));
            curandGenerateNormal(rng, temp_w, RNNWgtTensorSz, 0.0f, stddev);

            details::float2halfArr(reinterpret_cast<float*>(temp_w), reinterpret_cast<half1*>(tensors.w), RNNWgtTensorSz);

            cudaFree(temp_w);
        }
        if(std::is_same<T_NN, float>::value)  curandGenerateNormal(rng, reinterpret_cast<float*>(tensors.w), RNNWgtTensorSz, 0.0f, stddev);
        if(std::is_same<T_NN, double>::value) curandGenerateNormalDouble(rng, reinterpret_cast<double*>(tensors.w), RNNWgtTensorSz, 0.0, stddev);

        // init random dense weights and biases
        double stddevDense = sqrt(2. / (options.outVecSize + options.hiddenSize)); // glorot_uniform like tensorflow

        if(std::is_same<T_NN, half1>::value)  
        {
            float* temp_wDense;
            cudaMalloc(&temp_wDense, denseWgtTensorSz * sizeof(float));
            curandGenerateNormal(rng, temp_wDense, denseWgtTensorSz, 0.0f, stddev);

            details::float2halfArr(reinterpret_cast<float*>(temp_wDense), reinterpret_cast<half1*>(tensors.wDense), denseWgtTensorSz);
            
            cudaFree(temp_wDense);
        }
        if(std::is_same<T_NN, float>::value)  curandGenerateNormal(rng, reinterpret_cast<float*>(tensors.wDense), denseWgtTensorSz, 0.0f, stddevDense);
        if(std::is_same<T_NN, double>::value) curandGenerateNormalDouble(rng, reinterpret_cast<double*>(tensors.wDense), denseWgtTensorSz, 0.0, stddevDense);

        // all miniBatchSz dense layers have the same w
        cloneDevArr(tensors.wDense, options.hiddenSize * options.outVecSize, options.miniBatchSz * options.outLength); 

        // setup work space and reserved memory
        cudnnStatus_t stat = cudnnGetRNNTempSpaceSizes(cudnnHandle,
                                                       descriptors.rnn,
                                                       CUDNN_FWD_MODE_TRAINING,
                                                       descriptors.x,
                                                       &workSpaceSize,
                                                       &reserveSpaceSize);

        if(stat == CUDNN_STATUS_NOT_SUPPORTED && spOptions.algorithm == CUDNN_RNN_ALGO_PERSIST_STATIC) {
            throw std::invalid_argument("Hidden size is too large for a static persistent algorithm.");
        }

        cudaMalloc(&workSpace, workSpaceSize);
        cudaMalloc(&reserveSpace, reserveSpaceSize);

        if(spOptions.algorithm == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
            // Note: This step is expensive. Once completed the plan can be reused so long as the descriptor
            //       miniBatchSz or datatype don't change.
            cudnnBuildRNNDynamic(cudnnHandle, descriptors.rnn, options.miniBatchSz);
        }

    } catch ( std::exception &err ) {
        std::cerr << err.what() << std::endl;
        exit(-1);
    }
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::setSpOptions()
{
    spOptions.dataType       = getcuDNNDataType<T_NN>();
    spOptions.cuBLASDataType = getcuBLASDataType<T_NN>();

    switch(options.inputMode) {
        case InputMode::Skip:   spOptions.inputMode = CUDNN_SKIP_INPUT; break;
        case InputMode::Linear: spOptions.inputMode = CUDNN_LINEAR_INPUT; break;
    }
  
    switch(options.cellMode) {
        case CellMode::ReLU: spOptions.cellMode = CUDNN_RNN_RELU; break;
        case CellMode::Tanh: spOptions.cellMode = CUDNN_RNN_TANH; break;
        case CellMode::LSTM: spOptions.cellMode = CUDNN_LSTM; break;
        case CellMode::GRU:  spOptions.cellMode = CUDNN_GRU; break;
    }
  
    switch(options.biasMode) {
        case BiasMode::None:            spOptions.biasMode = CUDNN_RNN_NO_BIAS; break;
        case BiasMode::SingleInput:     spOptions.biasMode = CUDNN_RNN_SINGLE_INP_BIAS; break;
        case BiasMode::SingleRecurrent: spOptions.biasMode = CUDNN_RNN_SINGLE_REC_BIAS; break;
        case BiasMode::Double:          spOptions.biasMode = CUDNN_RNN_DOUBLE_BIAS; break;
    }
  
    switch(options.algorithm) {
        case Algorithm::Standard:       spOptions.algorithm = CUDNN_RNN_ALGO_STANDARD; break;
        case Algorithm::PersistStatic:  spOptions.algorithm = CUDNN_RNN_ALGO_PERSIST_STATIC; break;
        case Algorithm::PersistDynamic: spOptions.algorithm = CUDNN_RNN_ALGO_PERSIST_DYNAMIC; break;
    }
  
    switch(options.mathPrecision) {
        case MathPrecision::Half:   spOptions.mathPrecision = CUDNN_DATA_HALF; break;
        case MathPrecision::Float:  spOptions.mathPrecision = CUDNN_DATA_FLOAT; break;
        case MathPrecision::Double: spOptions.mathPrecision = CUDNN_DATA_DOUBLE; break;
    }
  
    switch(options.mathType) {
        case MathType::Default:        
            spOptions.mathType       = CUDNN_DEFAULT_MATH;
            spOptions.cuBLASMathType = CUBLAS_GEMM_DEFAULT; 
            break;
        case MathType::Tensorcore:     
            spOptions.mathType       = CUDNN_TENSOR_OP_MATH; 
            spOptions.cuBLASMathType = CUBLAS_GEMM_DEFAULT_TENSOR_OP; 
            break;
        case MathType::TensorcoreConv: 
            spOptions.mathType       = CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION; 
            spOptions.cuBLASMathType = CUBLAS_GEMM_DEFAULT_TENSOR_OP; 
            break;
    }
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::setOptions(RNNOptions_t extOptions)
{
    options = extOptions;

    try {
        checkOptionErr();
    } catch ( std::invalid_argument &err ) {
        std::cerr << err.what() << std::endl;
        exit(-1);
    }

    setSpOptions();
}



template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::allocTensors()
{
    inTensorSz        = options.seqLength * options.miniBatchSz * options.inVecSize; // seqLength and not inLength so that cuDNN doesn't complain (fill with 0)
    hidTensorSz       = options.numLayers * options.miniBatchSz * options.hiddenSize;
    hidOutTensorSz    = options.seqLength * options.miniBatchSz * options.hiddenSize; // from cuDNN library (y)
    tmpHidOutTensorSz = options.outLength * options.miniBatchSz * options.hiddenSize; // for y_temp in cuBLAS format (column-major and without inLength)
    outTensorSz       = options.outLength * options.miniBatchSz * options.outVecSize; // for the added dense layer
    denseWgtTensorSz  = options.outLength * options.miniBatchSz * options.hiddenSize * options.outVecSize;

    details::checkGPUAlloc( cudaMalloc(&tensors.x, inTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.y, hidOutTensorSz * sizeof(T_NN)) );

    details::checkGPUAlloc( cudaMalloc(&tensors.dx, inTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.dy, hidOutTensorSz * sizeof(T_NN)) );

    details::checkGPUAlloc( cudaMalloc(&tensors.hx, hidTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.cx, hidTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.hy, hidTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.cy, hidTensorSz * sizeof(T_NN)) );

    details::checkGPUAlloc( cudaMalloc(&tensors.dhx, hidTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.dcx, hidTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.dhy, hidTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.dcy, hidTensorSz * sizeof(T_NN)) );

    details::checkGPUAlloc( cudaMalloc(&tensors.y_temp, tmpHidOutTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.out, outTensorSz * sizeof(T_NN)) );

    details::checkGPUAlloc( cudaMalloc(&tensors.dy_temp, tmpHidOutTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.dout, outTensorSz * sizeof(T_NN)) );

    details::checkGPUAlloc( cudaMalloc(&tensors.wDense, denseWgtTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.dwDense, denseWgtTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.w, RNNWgtTensorSz * sizeof(T_NN)) );
    details::checkGPUAlloc( cudaMalloc(&tensors.dw, RNNWgtTensorSz * sizeof(T_NN)) );

    
    cudaMemset(tensors.hx, 0, hidTensorSz * sizeof(T_NN));
    cudaMemset(tensors.cx, 0, hidTensorSz * sizeof(T_NN));
    cudaMemset(tensors.hy, 0, hidTensorSz * sizeof(T_NN));
    cudaMemset(tensors.cy, 0, hidTensorSz * sizeof(T_NN));

    cudaMemset(tensors.dhx, 0, hidTensorSz * sizeof(T_NN));
    cudaMemset(tensors.dcx, 0, hidTensorSz * sizeof(T_NN));
    cudaMemset(tensors.dhy, 0, hidTensorSz * sizeof(T_NN));
    cudaMemset(tensors.dcy, 0, hidTensorSz * sizeof(T_NN));
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::setDescriptors()
{
    // aux vars for function calls
    float paddingFill = 0.0;
    seqLengthArray = new int[options.miniBatchSz];
    thrust::fill(thrust::host, seqLengthArray, seqLengthArray + options.miniBatchSz, options.seqLength);

    cudaMalloc(&devSeqLengthArray, options.miniBatchSz * sizeof(int));
    cudaMemcpy(devSeqLengthArray, seqLengthArray, options.miniBatchSz * sizeof(int), cudaMemcpyHostToDevice);
    
    // setup input and (hidden) output tensors
    cudnnCreateRNNDataDescriptor(&descriptors.x);
    cudnnCreateRNNDataDescriptor(&descriptors.y);
    
    cudnnSetRNNDataDescriptor(descriptors.x,
                              spOptions.dataType,
                              CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                              options.seqLength,
                              options.miniBatchSz,
                              options.inVecSize,
                              seqLengthArray,
                              &paddingFill);

    cudnnSetRNNDataDescriptor(descriptors.y,
                              spOptions.dataType,
                              CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                              options.seqLength,
                              options.miniBatchSz,
                              options.hiddenSize,
                              seqLengthArray,
                              &paddingFill);

    // setup hidden tensors
    cudnnCreateTensorDescriptor(&descriptors.h);
    cudnnCreateTensorDescriptor(&descriptors.c);
    cudnnSetTensorNdDescriptor(descriptors.h, spOptions.dataType, 3, dimHidden, strideHidden);
    cudnnSetTensorNdDescriptor(descriptors.c, spOptions.dataType, 3, dimHidden, strideHidden);

    // setup dropout descriptor
    cudnnCreateDropoutDescriptor(&descriptors.dropout);

    unsigned long long seed = 123ULL;
    size_t stateSize;

    cudnnDropoutGetStatesSize(cudnnHandle, &stateSize); 
    cudaMalloc(&states, stateSize); // These states are used to generate random numbers internally

    cudnnSetDropoutDescriptor(descriptors.dropout,
                              cudnnHandle,
                              options.dropout,
                              states,
                              stateSize,
                              seed);

    
    // setup RNN descriptor
    cudnnCreateRNNDescriptor(&descriptors.rnn);
    int projSize = options.hiddenSize; // no recurrent projection

    cudnnSetRNNDescriptor_v8(descriptors.rnn,
                             spOptions.algorithm,
                             spOptions.cellMode,
                             spOptions.biasMode,
                             CUDNN_UNIDIRECTIONAL,
                             spOptions.inputMode,
                             spOptions.dataType,
                             spOptions.mathPrecision,
                             spOptions.mathType,
                             options.inVecSize,
                             options.hiddenSize,
                             projSize,
                             options.numLayers,
                             descriptors.dropout,
                             0);

    // setup weight descriptors (only useful for accessing individual matrices)                         
    cudnnCreateTensorDescriptor(&descriptors.w);
    cudnnCreateTensorDescriptor(&descriptors.b);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::cloneDevArr(T_NN *devArr, int arrSz, int nClones)
{
   dim3 nThreads(256);
   dim3 nBlocks( (arrSz*(nClones-1) + nThreads.x - 1) / nThreads.x );

   details::cloneDevArr_ker<T_NN><<<nBlocks, nThreads>>>(devArr, arrSz, nClones);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::setInputs(std::vector<T_DATA>& inputs)
{
    dim3 nThreads(256);
    dim3 nBlocks( (inTensorSz + nThreads.x - 1) / nThreads.x );

    size_t nDataElems = inputs.size();

    nSequences = nDataElems/(options.inVecSize * options.inLength);
    nMiniBatches = nSequences/options.miniBatchSz;

    cudaMalloc(&devInputs, nDataElems * sizeof(T_DATA));
    cudaMemcpy(devInputs, &inputs[0], nDataElems * sizeof(T_DATA), cudaMemcpyHostToDevice);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::setTargets(std::vector<T_DATA>& targets)
{
    dim3 nThreads(256);
    dim3 nBlocks( (outTensorSz + nThreads.x - 1) / nThreads.x );

    size_t nDataElems = targets.size();

    cudaMalloc(&devTargets, nDataElems * sizeof(T_DATA));
    cudaMemcpy(devTargets, &targets[0], nDataElems * sizeof(T_DATA), cudaMemcpyHostToDevice);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::setSeed(unsigned long long extWgtSeed) {
    wgtSeed = extWgtSeed;
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::setMiniBatchInputs(int *devShuffledSeqIdx, int miniB_idx)
{
    dim3 nThreads(256);
    dim3 nBlocks( (inTensorSz + nThreads.x - 1) / nThreads.x );

    details::setMiniBatchInputs_ker<<<nBlocks, nThreads>>>(devInputs, tensors.x, inTensorSz, nSequences,
                                        options.inLength, options.seqLength, options.miniBatchSz, options.inVecSize, devShuffledSeqIdx, miniB_idx);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::translateLayer()
{
    int nElems = options.seqLength * options.miniBatchSz * options.hiddenSize;

    dim3 nThreads(256);
    dim3 nBlocks( (nElems + nThreads.x - 1) / nThreads.x );

    details::translateLayer_ker<T_NN><<<nBlocks, nThreads>>>(tensors.y, tensors.y_temp, 
                                        options.seqLength, options.outLength, options.miniBatchSz, options.hiddenSize, nElems);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::calc_dout(int *devShuffledSeqIdx, int miniB_idx)
{
    int nElems = options.outLength * options.miniBatchSz * options.outVecSize;

    dim3 nThreads(256);
    dim3 nBlocks( (nElems + nThreads.x - 1) / nThreads.x );

    details::calc_dout_ker<<<nBlocks, nThreads>>>(tensors.out, tensors.dout, devTargets, 
                                         options.miniBatchSz, options.outLength, options.outVecSize, devShuffledSeqIdx, miniB_idx, nSequences, nElems);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::calc_dy()
{
    int nElems = options.seqLength * options.miniBatchSz * options.hiddenSize;

    dim3 nThreads(256);
    dim3 nBlocks( (nElems + nThreads.x - 1) / nThreads.x );

    details::calc_dy_ker<T_NN><<<nBlocks, nThreads>>>(tensors.dy, tensors.dy_temp, options.hiddenSize, options.seqLength, options.outLength, options.miniBatchSz, nElems);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::calc_dwDense()
{
    int nWgts = options.hiddenSize * options.outVecSize;

    dim3 nThreads(256);
    dim3 nBlocks( (nWgts + nThreads.x - 1) / nThreads.x );

    details::calc_dwDense_ker<T_NN><<<nBlocks, nThreads>>>(tensors.dwDense, options.miniBatchSz, options.outLength, options.hiddenSize, nWgts);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::updateWgtsRNN_SGD()  
{
   dim3 nThreads(256);
   dim3 nBlocks( (RNNWgtTensorSz + nThreads.x - 1) / nThreads.x );

   details::updateWgtsRNN_SGD_ker<T_NN><<<nBlocks, nThreads>>>(tensors.w, tensors.dw, optimOpts.lr, RNNWgtTensorSz);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::updateWgtsDense_SGD()
{
    dim3 nThreads(256);
    dim3 nBlocks( (denseWgtTensorSz + nThreads.x - 1) / nThreads.x );

    details::updateWgtsDense_SGD_ker<T_NN><<<nBlocks, nThreads>>>(tensors.wDense, tensors.dwDense, optimOpts.lr, denseWgtTensorSz);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::updateWgtsRNN_RMSProp()
{
   dim3 nThreads(256);
   dim3 nBlocks( (RNNWgtTensorSz + nThreads.x - 1) / nThreads.x );

   details::updateWgtsRNN_RMSProp_ker<T_NN><<<nBlocks, nThreads>>>(tensors.w, tensors.dw, vRNN, optimOpts.lr, optimOpts.gamma, RNNWgtTensorSz);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::updateWgtsDense_RMSProp()
{
    int nWgts = options.hiddenSize * options.outVecSize;

    dim3 nThreads(256);
    dim3 nBlocks( (denseWgtTensorSz + nThreads.x - 1) / nThreads.x );

    details::updateWgtsDense_RMSProp_ker<T_NN><<<nBlocks, nThreads>>>(tensors.wDense, tensors.dwDense, vDense, optimOpts.lr, optimOpts.gamma, denseWgtTensorSz, nWgts);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::updateWgtsRNN_Adam()
{  
    dim3 nThreads(256);
    dim3 nBlocks( (RNNWgtTensorSz + nThreads.x - 1) / nThreads.x );

    details::updateWgtsRNN_Adam_ker<T_NN><<<nBlocks, nThreads>>>(tensors.w, tensors.dw, mRNN, vRNN, 
                                                        optimOpts.lr, optimOpts.b1, optimOpts.b2, b1t, b2t, optimOpts.epsilon,
                                                        RNNWgtTensorSz);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::updateWgtsDense_Adam()
{
    int nWgts = options.hiddenSize * options.outVecSize;
    
    dim3 nThreads(256);
    dim3 nBlocks( (denseWgtTensorSz + nThreads.x - 1) / nThreads.x );

    details::updateWgtsDense_Adam_ker<T_NN><<<nBlocks, nThreads>>>(tensors.wDense, tensors.dwDense, mDense, vDense, 
                                                          optimOpts.lr, optimOpts.b1, optimOpts.b2, b1t, b2t, optimOpts.epsilon, 
                                                          denseWgtTensorSz, nWgts);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::train(int epochs, bool shuffleSequences, bool showProgress)
{
    int *shuffledSeqIdx = new int[nSequences];
    thrust::sequence(thrust::host, shuffledSeqIdx, shuffledSeqIdx + nSequences, 0);

    int *devShuffledSeqIdx;
    cudaMalloc(&devShuffledSeqIdx, sizeof(int)*nSequences);
    cudaMemcpy(devShuffledSeqIdx, shuffledSeqIdx, sizeof(int)*nSequences, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float time;

    if(showProgress)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    for(int i = 0; i < epochs; ++i)
    {
        if(shuffleSequences) {
            std::random_shuffle(shuffledSeqIdx, shuffledSeqIdx + nSequences);
            cudaMemcpy(devShuffledSeqIdx, shuffledSeqIdx, sizeof(int)*nSequences, cudaMemcpyHostToDevice);
        }

        for(int j = 0; j < nMiniBatches; ++j)
        {
            setMiniBatchInputs(devShuffledSeqIdx, j);

            cudnnRNNForward(cudnnHandle,
                            descriptors.rnn,
                            CUDNN_FWD_MODE_TRAINING,
                            devSeqLengthArray,
                            descriptors.x,
                            tensors.x,
                            descriptors.y,
                            tensors.y,
                            descriptors.h,
                            tensors.hx,
                            tensors.hy,
                            descriptors.c,
                            tensors.cx,
                            tensors.cy,
                            wgtSpaceByteSize,
                            tensors.w,
                            workSpaceSize,
                            workSpace,
                            reserveSpaceSize,
                            reserveSpace);

            /*** dense feedforward ***/
            translateLayer();

            cublasGemmStridedBatchedEx(cublasHandle, 
                                       CUBLAS_OP_N, 
                                       CUBLAS_OP_N, 
                                       options.outVecSize, 
                                       1, 
                                       options.hiddenSize, 
                                       &alpha, 
                                       tensors.wDense, 
                                       spOptions.cuBLASDataType, 
                                       options.outVecSize, 
                                       options.outVecSize*options.hiddenSize, 
                                       tensors.y_temp, 
                                       spOptions.cuBLASDataType, 
                                       options.hiddenSize, 
                                       options.hiddenSize, 
                                       &beta, 
                                       tensors.out, 
                                       spOptions.cuBLASDataType, 
                                       options.outVecSize, 
                                       options.outVecSize, 
                                       options.miniBatchSz*options.outLength, 
                                       spOptions.cuBLASDataType,
                                       spOptions.cuBLASMathType);
            /*************************/

            /*** calculate dy (truncated backprop) ***/
            calc_dout(devShuffledSeqIdx, j);

            cublasGemmStridedBatchedEx(cublasHandle, 
                                       CUBLAS_OP_T, 
                                       CUBLAS_OP_N, 
                                       options.hiddenSize, 
                                       1, 
                                       options.outVecSize, 
                                       &alpha, 
                                       tensors.wDense, 
                                       spOptions.cuBLASDataType, 
                                       options.outVecSize, 
                                       options.outVecSize*options.hiddenSize, 
                                       tensors.dout, 
                                       spOptions.cuBLASDataType, 
                                       options.outVecSize, 
                                       options.outVecSize, 
                                       &beta, 
                                       tensors.dy_temp, 
                                       spOptions.cuBLASDataType, 
                                       options.hiddenSize, 
                                       options.hiddenSize, 
                                       options.miniBatchSz*options.outLength, 
                                       spOptions.cuBLASDataType, 
                                       spOptions.cuBLASMathType);

            calc_dy();
            /*****************************************/

            /*** calculate dense weights grad ***/
            cublasGemmStridedBatchedEx(cublasHandle, 
                                       CUBLAS_OP_N, 
                                       CUBLAS_OP_T, 
                                       options.outVecSize, 
                                       options.hiddenSize, 
                                       1, 
                                       &alpha, 
                                       tensors.dout, 
                                       spOptions.cuBLASDataType, 
                                       options.outVecSize, 
                                       options.outVecSize, 
                                       tensors.y_temp, 
                                       spOptions.cuBLASDataType, 
                                       options.hiddenSize, 
                                       options.hiddenSize, 
                                       &beta, 
                                       tensors.dwDense, 
                                       spOptions.cuBLASDataType, 
                                       options.outVecSize, 
                                       options.hiddenSize*options.outVecSize, 
                                       options.miniBatchSz*options.outLength, 
                                       spOptions.cuBLASDataType, 
                                       spOptions.cuBLASMathType);

            calc_dwDense();
            /**********************************/

            cudnnRNNBackwardData_v8(cudnnHandle,
                                    descriptors.rnn,
                                    devSeqLengthArray,
                                    descriptors.y,
                                    tensors.y,
                                    tensors.dy,
                                    descriptors.x,
                                    tensors.dx,
                                    descriptors.h,
                                    tensors.hx,
                                    tensors.dhy,
                                    tensors.dhx,
                                    descriptors.c,
                                    tensors.cx,
                                    tensors.dcy,
                                    tensors.dcx,
                                    wgtSpaceByteSize,
                                    tensors.w,
                                    workSpaceSize,
                                    workSpace,
                                    reserveSpaceSize,
                                    reserveSpace);
               
            // cudnnRNNBackwardWeights adds to the data in dw.
            cudaMemset(tensors.dw, 0, wgtSpaceByteSize);
         
            cudnnRNNBackwardWeights_v8(cudnnHandle,
                                       descriptors.rnn,
                                       CUDNN_WGRAD_MODE_ADD,
                                       devSeqLengthArray,
                                       descriptors.x,
                                       tensors.x,
                                       descriptors.h,
                                       tensors.hx,
                                       descriptors.y,
                                       tensors.y,
                                       wgtSpaceByteSize,
                                       tensors.dw,
                                       workSpaceSize,
                                       workSpace,
                                       reserveSpaceSize,
                                       reserveSpace);

            /*** update weights ***/
            switch(optimOpts.optimizer)
            {
                case cudaRNN::Optimizer::SGD:
                    updateWgtsDense_SGD();
                    updateWgtsRNN_SGD();
                    break;
                case cudaRNN::Optimizer::RMSProp:
                    updateWgtsDense_RMSProp();
                    updateWgtsRNN_RMSProp();
                    break;
                case cudaRNN::Optimizer::Adam:
                    b1t *= optimOpts.b1;
                    b2t *= optimOpts.b2;
                    updateWgtsDense_Adam();
                    updateWgtsRNN_Adam();
                    break;
                default: // SGD
                    updateWgtsDense_SGD();
                    updateWgtsRNN_SGD();
                    break;
            }
            /**********************/

            /*** calc loss ***/
            if(metricType != cudaRNN::MetricType::None && j % nIterToCalcLoss == 0) 
            {
                float loss = calc_loss();

                int step = details::idx_2Dto1D(i, j, nMiniBatches);
                printLoss(step, loss);
            }
            /*****************/

            if(showProgress) {
                // cudaDeviceSynchronize();
                std::cout << "\rProgress " << 100*details::idx_2Dto1D(i, j, nMiniBatches)/(nMiniBatches*epochs) << "%" << std::flush;
            }
        }
    }

    if(showProgress)
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&time, start, stop);

        std::cout << "\rDone!\t\t" << std::endl;

        std::cout << std::endl << "Total training time: " << time/1000 << " s" << std::endl
                               << "Time per iteration: "  << time/(nMiniBatches*epochs) << " ms";
    }

    cudaFree(devShuffledSeqIdx);
    delete [] shuffledSeqIdx;
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::RNNSinglePassExample()
{
    cudnnRNNForward(cudnnHandle,
                    descriptors.rnn,
                    CUDNN_FWD_MODE_TRAINING,
                    devSeqLengthArray,
                    descriptors.x,
                    tensors.x,
                    descriptors.y,
                    tensors.y,
                    descriptors.h,
                    tensors.hx,
                    tensors.hy,
                    descriptors.c,
                    tensors.cx,
                    tensors.cy,
                    wgtSpaceByteSize,
                    tensors.w,
                    workSpaceSize,
                    workSpace,
                    reserveSpaceSize,
                    reserveSpace);

    cudnnRNNBackwardData_v8(cudnnHandle,
                            descriptors.rnn,
                            devSeqLengthArray,
                            descriptors.y,
                            tensors.y,
                            tensors.dy,
                            descriptors.x,
                            tensors.dx,
                            descriptors.h,
                            tensors.hx,
                            tensors.dhy,
                            tensors.dhx,
                            descriptors.c,
                            tensors.cx,
                            tensors.dcy,
                            tensors.dcx,
                            wgtSpaceByteSize,
                            tensors.w,
                            workSpaceSize,
                            workSpace,
                            reserveSpaceSize,
                            reserveSpace);
        
    // cudnnRNNBackwardWeights adds to the data in dw.
    cudaMemset(tensors.dw, 0, wgtSpaceByteSize);
    
    cudnnRNNBackwardWeights_v8(cudnnHandle,
                               descriptors.rnn,
                               CUDNN_WGRAD_MODE_ADD,
                               devSeqLengthArray,
                               descriptors.x,
                               tensors.x,
                               descriptors.h,
                               tensors.hx,
                               descriptors.y,
                               tensors.y,
                               wgtSpaceByteSize,
                               tensors.dw,
                               workSpaceSize,
                               workSpace,
                               reserveSpaceSize,
                               reserveSpace);
    
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::test(int epochs, bool printOutputs, std::string outputsFilename)
{
    static std::ofstream out;
    if(printOutputs) out.open(outputsFilename); 

    int *seqIdx = new int[nSequences];
    thrust::sequence(thrust::host, seqIdx, seqIdx + nSequences, 0);

    int *devSeqIdx;
    cudaMalloc(&devSeqIdx, sizeof(int)*nSequences);

    cudaMemcpy(devSeqIdx, seqIdx, sizeof(int)*nSequences, cudaMemcpyHostToDevice);

    for(int i = 0; i < epochs; ++i)
    {
        for(int j = 0; j < nMiniBatches; ++j)
        {
            setMiniBatchInputs(devSeqIdx, j);

            cudnnRNNForward(cudnnHandle,
                            descriptors.rnn,
                            CUDNN_FWD_MODE_INFERENCE,
                            devSeqLengthArray,
                            descriptors.x,
                            tensors.x,
                            descriptors.y,
                            tensors.y,
                            descriptors.h,
                            tensors.hx,
                            tensors.hy,
                            descriptors.c,
                            tensors.cx,
                            tensors.cy,
                            wgtSpaceByteSize,
                            tensors.w,
                            workSpaceSize,
                            workSpace,
                            reserveSpaceSize,
                            reserveSpace);

            /*** dense feedforward ***/
            translateLayer();

            cublasGemmStridedBatchedEx(cublasHandle, 
                                       CUBLAS_OP_N, 
                                       CUBLAS_OP_N, 
                                       options.outVecSize, 
                                       1, 
                                       options.hiddenSize, 
                                       &alpha, 
                                       tensors.wDense, 
                                       spOptions.cuBLASDataType, 
                                       options.outVecSize, 
                                       options.outVecSize*options.hiddenSize, 
                                       tensors.y_temp, 
                                       spOptions.cuBLASDataType, 
                                       options.hiddenSize, 
                                       options.hiddenSize, 
                                       &beta, 
                                       tensors.out, 
                                       spOptions.cuBLASDataType, 
                                       options.outVecSize, 
                                       options.outVecSize, 
                                       options.miniBatchSz*options.outLength, 
                                       spOptions.cuBLASDataType, 
                                       spOptions.cuBLASMathType);
            /*************************/

            /*** calc loss ***/
            if(metricType != cudaRNN::MetricType::None && j % nIterToCalcLoss == 0) 
            {
                float loss = calc_loss();

                int step = details::idx_2Dto1D(i, j, nMiniBatches);
                printLoss(step, loss);
            }
            /*****************/

            if(printOutputs)
            {
                T_NN *hOut = new T_NN[outTensorSz];
                cudaMemcpy(hOut, tensors.out, outTensorSz * sizeof(T_NN), cudaMemcpyDeviceToHost);                
                // options.outVecSize * options.miniBatchSz * options.outLength;
                for(int i = 0; i < options.miniBatchSz; ++i)
                {
                    for(int j = 0; j < options.outLength; ++j)
                    {
                        for(int k = 0; k < options.outVecSize; ++k)
                            out << hOut[details::idx_3Dto1D(k, i, j, options.miniBatchSz, options.outLength)] << " ";
                        out << std::endl;
                    }
                    out << std::endl << std::endl;           
                }
                delete [] hOut;
                
            }
        }
    }

    if(printOutputs) out.close();

    cudaFree(devSeqIdx);
    delete [] seqIdx;
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::setOptimizer(optimizerOptions_t extOptimOpts)
{
    optimOpts = extOptimOpts;

    switch(optimOpts.optimizer)
    {   
        case cudaRNN::Optimizer::RMSProp: // RMSProp
            cudaMalloc(&vRNN, RNNWgtTensorSz * sizeof(T_NN));
            cudaMalloc(&vDense, options.hiddenSize * options.outVecSize * sizeof(T_NN));

            cudaMemset(vRNN, 0, RNNWgtTensorSz * sizeof(T_NN));
            cudaMemset(vDense, 0, options.hiddenSize * options.outVecSize * sizeof(T_NN));

            break;
        case cudaRNN::Optimizer::Adam: // Adam
            cudaMalloc(&vRNN, RNNWgtTensorSz * sizeof(T_NN));
            cudaMalloc(&mRNN, RNNWgtTensorSz * sizeof(T_NN));
            cudaMalloc(&vDense, options.hiddenSize * options.outVecSize * sizeof(T_NN));
            cudaMalloc(&mDense, options.hiddenSize * options.outVecSize * sizeof(T_NN));

            cudaMemset(vRNN, 0, RNNWgtTensorSz * sizeof(T_NN));
            cudaMemset(mRNN, 0, RNNWgtTensorSz * sizeof(T_NN));
            cudaMemset(vDense, 0, options.hiddenSize * options.outVecSize * sizeof(T_NN));
            cudaMemset(mDense, 0, options.hiddenSize * options.outVecSize * sizeof(T_NN));

            break;
        default: // SGD
            break; // alloc not needed
    }
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::save(std::string wgtsFilename, std::string optsFilename)
{
    std::ofstream saveFile; 

    saveFile.open(optsFilename);
    saveFile << options << std::endl;
    saveFile.close();

    T_NN *hRNNWgtTensor = new T_NN[RNNWgtTensorSz];
    cudaMemcpy(hRNNWgtTensor, tensors.w, RNNWgtTensorSz * sizeof(T_NN), cudaMemcpyDeviceToHost);
    T_NN *hDenseWgtTensor = new T_NN[denseWgtTensorSz];
    cudaMemcpy(hDenseWgtTensor, tensors.wDense, denseWgtTensorSz * sizeof(T_NN), cudaMemcpyDeviceToHost);

    saveFile.open(wgtsFilename);
    for(int i = 0; i < RNNWgtTensorSz; ++i)
        saveFile << hRNNWgtTensor[i] << " ";

    saveFile << std::endl;

    for(int i = 0; i < denseWgtTensorSz; ++i)
        saveFile << hDenseWgtTensor[i] << " ";

    delete [] hRNNWgtTensor;
    delete [] hDenseWgtTensor;

    saveFile.close();
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::load(std::string wgtsFilename, std::string optsFilename)
{
    std::ifstream loadFile; 

    loadFile.open(optsFilename);
    loadFile >> options;
    loadFile.close();

    // cleanup
    freeTensors();
    destroyDescriptors();

    delete [] seqLengthArray;
    cudaFree(workSpace);
    cudaFree(reserveSpace);
    cudaFree(states);
    cudaFree(devSeqLengthArray);

    try {
        checkOptionErr();
        setSpOptions();    

        dimHidden[0] = options.numLayers;
        dimHidden[1] = options.miniBatchSz;
        dimHidden[2] = options.hiddenSize;
        strideHidden[0] = dimHidden[1] * dimHidden[2];
        strideHidden[1] = dimHidden[2];
        strideHidden[2] = 1;

        setDescriptors();

        cudnnGetRNNWeightSpaceSize(cudnnHandle, descriptors.rnn, &wgtSpaceByteSize);
        RNNWgtTensorSz = wgtSpaceByteSize/sizeof(T_NN);

        allocTensors();

        // load saved weights
        T_NN *hRNNWgtTensor = new T_NN[RNNWgtTensorSz];
        T_NN *hDenseWgtTensor = new T_NN[denseWgtTensorSz];
    
        loadFile.open(wgtsFilename);
        for(int i = 0; i < RNNWgtTensorSz; ++i)
           loadFile >> hRNNWgtTensor[i];
    
        for(int i = 0; i < denseWgtTensorSz; ++i)
           loadFile >> hDenseWgtTensor[i];
    
        cudaError_t cudaStat;
        cudaStat = cudaMemcpy(tensors.w, hRNNWgtTensor, RNNWgtTensorSz * sizeof(T_NN), cudaMemcpyHostToDevice);
        cudaStat = cudaMemcpy(tensors.wDense, hDenseWgtTensor, denseWgtTensorSz * sizeof(T_NN), cudaMemcpyHostToDevice);
        if(cudaStat != CUDA_SUCCESS) throw std::bad_alloc();
    
        delete [] hRNNWgtTensor;
        delete [] hDenseWgtTensor;
    
        loadFile.close();
        
        // setup work space and reserved memory
        cudnnStatus_t cudnnStat = cudnnGetRNNTempSpaceSizes(cudnnHandle,
                                                            descriptors.rnn,
                                                            CUDNN_FWD_MODE_TRAINING,
                                                            descriptors.x,
                                                            &workSpaceSize,
                                                            &reserveSpaceSize);

        if(cudnnStat == CUDNN_STATUS_NOT_SUPPORTED && spOptions.algorithm == CUDNN_RNN_ALGO_PERSIST_STATIC) {
            throw std::invalid_argument("Hidden size is too large for a static persistent algorithm.");
        }

        cudaMalloc(&workSpace, workSpaceSize);
        cudaMalloc(&reserveSpace, reserveSpaceSize);

        if(spOptions.algorithm == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
            // Note: This step is expensive. Once completed the plan can be reused so long as the descriptor
            //       miniBatchSz or datatype don't change.
            cudnnBuildRNNDynamic(cudnnHandle, descriptors.rnn, options.miniBatchSz);
        }

    } catch ( std::exception &err ) {
        std::cerr << err.what() << std::endl;
        exit(-1);
    }
}

template <typename T_DATA, typename T_NN> float cudaRNN::RNN<T_DATA, T_NN>::calc_loss()
{
    dim3 nThreads(256);
    dim3 nBlocks( (outTensorSz + nThreads.x - 1) / nThreads.x );

    float loss = 0;
    float *devLoss;
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    switch(metricType)
    {
        case cudaRNN::MetricType::MSE: // mean sq err
            details::squareElems_ker<T_NN><<<nBlocks, nThreads>>>(tensors.dout, outTensorSz);

            // Declare, allocate, and initialize device-accessible pointers for input and output
            cudaMalloc(&devLoss, sizeof(float));
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tensors.dout, devLoss, outTensorSz);
            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            // Run sum-reduction
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tensors.dout, devLoss, outTensorSz);

            cudaMemcpy(&loss, devLoss, sizeof(float), cudaMemcpyDeviceToHost);
            return sqrtf(loss/outTensorSz);

        case cudaRNN::MetricType::MAE: // mean abs err
            details::absValElems_ker<T_NN><<<nBlocks, nThreads>>>(tensors.dout, outTensorSz);

            // Declare, allocate, and initialize device-accessible pointers for input and output
            cudaMalloc(&devLoss, sizeof(float));
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tensors.dout, devLoss, outTensorSz);
            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            // Run sum-reduction
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, tensors.dout, devLoss, outTensorSz);

            cudaMemcpy(&loss, devLoss, sizeof(float), cudaMemcpyDeviceToHost);
            return loss/outTensorSz;

        default:
            break;
    }
    return 0;
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::printLoss(int step, float loss)
{
    static std::ofstream out(lossFilename);

    static float meanLoss = 0;
    static int i = 1;

    meanLoss += 1./i * (loss - meanLoss);
    ++i;

    out << step << " " << meanLoss << std::endl;
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::setMetrics(cudaRNN::MetricType extMetricType, int extNIterToCalcLoss, std::string extLossFilename)
{
    metricType = extMetricType;
    nIterToCalcLoss = extNIterToCalcLoss;
    lossFilename = extLossFilename;
}

template <typename T_DATA, typename T_NN> cudaRNN::RNN<T_DATA, T_NN>::~RNN()
{
    destroyDescriptors();
    freeTensors();

    switch(optimOpts.optimizer)
    {   
        case cudaRNN::Optimizer::RMSProp:
            cudaFree(vRNN);
            cudaFree(vDense);
            break;
        case cudaRNN::Optimizer::Adam:
            cudaFree(vRNN);
            cudaFree(mRNN);
            cudaFree(vDense);
            cudaFree(mDense);
            break;
        default: // SGD
            break; // alloc not needed
    }

    delete [] seqLengthArray;
    cudaFree(workSpace);
    cudaFree(reserveSpace);
    cudaFree(states);
    cudaFree(devSeqLengthArray);

    cudnnDestroy(cudnnHandle);
    cublasDestroy(cublasHandle);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::destroyDescriptors()
{
    cudnnDestroyRNNDataDescriptor(descriptors.x);
    cudnnDestroyRNNDataDescriptor(descriptors.y);

    cudnnDestroyTensorDescriptor(descriptors.h);
    cudnnDestroyTensorDescriptor(descriptors.c);

    cudnnDestroyTensorDescriptor(descriptors.w);
    cudnnDestroyTensorDescriptor(descriptors.b);

    cudnnDestroyDropoutDescriptor(descriptors.dropout);
    cudnnDestroyRNNDescriptor(descriptors.rnn);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::freeTensors()
{
    cudaFree(tensors.x);
    cudaFree(tensors.y);

    cudaFree(tensors.dx);
    cudaFree(tensors.dy);

    cudaFree(tensors.hx);
    cudaFree(tensors.cx);
    cudaFree(tensors.hy);
    cudaFree(tensors.cy);

    cudaFree(tensors.dhx);
    cudaFree(tensors.dcx);
    cudaFree(tensors.dhy);
    cudaFree(tensors.dcy);

    cudaFree(tensors.y_temp);
    cudaFree(tensors.out);

    cudaFree(tensors.dy_temp);
    cudaFree(tensors.dout);

    cudaFree(tensors.wDense);
    cudaFree(tensors.dwDense);
    cudaFree(tensors.w);
    cudaFree(tensors.dw);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::initGPUExampleData(T_NN *data, int nElems, T_NN value) 
{
    dim3 nThreads(256);
    dim3 nBlocks( (nElems + nThreads.x - 1) / nThreads.x );

    details::initGPUExampleData_ker<T_NN><<<nBlocks, nThreads>>>(data, nElems, value);
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::initDataExample()
{
    initGPUExampleData(tensors.x, inTensorSz, 1.0);
    initGPUExampleData(tensors.hx, hidTensorSz, 1.0);
    initGPUExampleData(tensors.cx, hidTensorSz, 1.0);

    initGPUExampleData(tensors.dy, outTensorSz, 1.0);
    initGPUExampleData(tensors.dhy, hidTensorSz, 1.0);
    initGPUExampleData(tensors.dcy, hidTensorSz, 1.0);

    int numLinearLayers = 0;
    if (spOptions.cellMode == CUDNN_RNN_RELU || spOptions.cellMode == CUDNN_RNN_TANH) {
        numLinearLayers = 2;
    } else if (spOptions.cellMode == CUDNN_LSTM) {
        numLinearLayers = 8;
    } else if (spOptions.cellMode == CUDNN_GRU) {
        numLinearLayers = 6;
    }

    for(int layer = 0; layer < options.numLayers; layer++) 
    {
        for(int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) 
        {
            cudnnDataType_t dataTypeTemp;
            int nDims = 0;
            int dim[3], stride[3];
            T_NN *linLayerMat  = NULL;
            T_NN *linLayerBias = NULL;

            cudnnGetRNNWeightParams(cudnnHandle,
                                    descriptors.rnn,
                                    layer,
                                    wgtSpaceByteSize,
                                    tensors.w,
                                    linLayerID,
                                    descriptors.w,
                                    (void **)&linLayerMat,
                                    descriptors.b,
                                    (void **)&linLayerBias);

            if(linLayerMat) {
                cudnnGetTensorNdDescriptor(descriptors.w, 3, &dataTypeTemp, &nDims, dim, stride);
                initGPUExampleData(linLayerMat , dim[0] * dim[1] * dim[2], 1.0 / (dim[0] * dim[1] * dim[2]));
            }
            if(linLayerBias) {
                cudnnGetTensorNdDescriptor(descriptors.b, 3, &dataTypeTemp, &nDims, dim, stride);
                initGPUExampleData(linLayerBias, dim[0] * dim[1] * dim[2], 1.0);
            }
        }
    }
}

template <typename T_DATA, typename T_NN> void cudaRNN::RNN<T_DATA, T_NN>::performChecksums()
{
    if (true) {
        T_NN *testOutputy;
        T_NN *testOutputhy;
        T_NN *testOutputcy;

        testOutputy = (T_NN *)malloc(hidOutTensorSz * sizeof(T_NN));
        testOutputhy = (T_NN *)malloc(hidTensorSz * sizeof(T_NN));
        testOutputcy = (T_NN *)malloc(hidTensorSz * sizeof(T_NN));

        cudaMemcpy(testOutputy, tensors.y, hidOutTensorSz * sizeof(T_NN), cudaMemcpyDeviceToHost);
        if (tensors.hy != NULL) {
            cudaMemcpy(testOutputhy, tensors.hy, hidTensorSz * sizeof(T_NN), cudaMemcpyDeviceToHost);
        }
        if (tensors.cy != NULL && spOptions.cellMode == CUDNN_LSTM) {
            cudaMemcpy(testOutputcy, tensors.cy, hidTensorSz * sizeof(T_NN), cudaMemcpyDeviceToHost);
        }

        double checksumy = 0.f;
        double checksumhy = 0.f;
        double checksumcy = 0.f;

        for (int m = 0; m < options.miniBatchSz; m++) {
            double localSumi = 0;
            double localSumh = 0;
            double localSumc = 0;

            for (int j = 0; j < options.seqLength; j++) {
                for (int i = 0; i < options.hiddenSize; i++) {
                    localSumi += (double) testOutputy[j * options.miniBatchSz * options.hiddenSize + m * options.hiddenSize + i];
                }
            }
            for (int j = 0; j < options.numLayers; j++) {
                for (int i = 0; i < options.hiddenSize; i++) {
                    if (tensors.hy != NULL) {
                        localSumh += (double) testOutputhy[j * options.hiddenSize * options.miniBatchSz + m * options.hiddenSize + i];
                    }
                    if ((tensors.cy != NULL) && (spOptions.cellMode == CUDNN_LSTM)) {
                        localSumc += (double) testOutputcy[j * options.hiddenSize * options.miniBatchSz + m * options.hiddenSize + i];
                    }
                }
            }

            checksumy += localSumi;
            checksumhy += localSumh;
            checksumcy += localSumc;
        }

        printf("y checksum %E     ", checksumy);
        if (spOptions.cellMode == CUDNN_LSTM) {
            printf("cy checksum %E     ", checksumcy);
        }
        printf("hy checksum %E\n", checksumhy);

        free(testOutputy);
        free(testOutputcy);
        free(testOutputhy);
    }

    if (true) {
        T_NN *testOutputdx;
        T_NN *testOutputdhx;
        T_NN *testOutputdcx;

        testOutputdx = (T_NN *)malloc(inTensorSz * sizeof(T_NN));
        testOutputdhx = (T_NN *)malloc(hidTensorSz * sizeof(T_NN));
        testOutputdcx = (T_NN *)malloc(hidTensorSz * sizeof(T_NN));

        cudaMemcpy(testOutputdx, tensors.dx, inTensorSz * sizeof(T_NN), cudaMemcpyDeviceToHost);
        if (tensors.dhx != NULL) {
            cudaMemcpy(testOutputdhx, tensors.dhx, hidTensorSz * sizeof(T_NN), cudaMemcpyDeviceToHost);
        }
        if ((tensors.dcx != NULL) && (spOptions.cellMode == CUDNN_LSTM)) {
            cudaMemcpy(testOutputdcx, tensors.dcx, hidTensorSz * sizeof(T_NN), cudaMemcpyDeviceToHost);
        }

        double checksumdx = 0.f;
        double checksumdhx = 0.f;
        double checksumdcx = 0.f;

        for (int m = 0; m < options.miniBatchSz; m++) {
            double localSumdx = 0;
            double localSumdhx = 0;
            double localSumdcx = 0;

            for (int j = 0; j < options.seqLength; j++) {
                for (int i = 0; i < options.inVecSize; i++) {
                    localSumdx += (double) testOutputdx[j * options.miniBatchSz * options.inVecSize + m * options.inVecSize + i];
                }
            }

            for (int j = 0; j < options.numLayers; j++) {
                for (int i = 0; i < options.hiddenSize; i++) {
                    localSumdhx += (double) testOutputdhx[j * options.hiddenSize * options.miniBatchSz + m * options.hiddenSize + i];
                    if(spOptions.cellMode == CUDNN_LSTM) {
                        localSumdcx += (double) testOutputdcx[j * options.hiddenSize * options.miniBatchSz + m * options.hiddenSize + i];
                    }
                }
            }

            checksumdx += localSumdx;
            checksumdhx += localSumdhx;
            checksumdcx += localSumdcx;
        }

        printf("dx checksum %E    ", checksumdx);
        if (spOptions.cellMode == CUDNN_LSTM) {
            printf("dcx checksum %E    ", checksumdcx);
        }
        printf("dhx checksum %E\n", checksumdhx);

        free(testOutputdx);
        free(testOutputdhx);
        free(testOutputdcx);
    }

    if (true) {
        T_NN *testOutputdw;
        testOutputdw = (T_NN *)malloc(wgtSpaceByteSize);

        cudaMemcpy(testOutputdw, tensors.dw, wgtSpaceByteSize, cudaMemcpyDeviceToHost);

        double checksumdw = 0.;

        for (int i = 0; i < RNNWgtTensorSz; i++) {
            checksumdw += (double) testOutputdw[i];
        }

        printf("dw checksum %E\n", checksumdw);

        free(testOutputdw);
    }
}