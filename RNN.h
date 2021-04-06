#pragma once

#include <cudnn.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>

namespace cudaRNN
{
    template <typename T_DATA, typename T_NN> class RNN;

    struct RNNOptions_t;
    struct optimizerOptions_t;

    namespace details
    {
        struct spRNNOptions_t;
        struct Descriptors_t;
        template <typename T_NN> struct Tensors_t;

        __host__ __device__ __inline__ int idx_2Dto1D(int x2, int x1, int D1) { return x2*D1 + x1; }
        __host__ __device__ __inline__ int idx_3Dto1D(int x3, int x2, int x1, int D2, int D1) { return x3*(D2*D1) + x2*D1 + x1; }
    }
    

    enum class Optimizer { SGD, RMSProp, Adam };
    enum class MetricType { None, MSE, MAE };

    enum class InputMode { Skip, Linear };
    enum class CellMode { ReLU, Tanh, LSTM, GRU };
    enum class BiasMode { None, SingleInput, SingleRecurrent, Double };
    enum class Algorithm { Standard, PersistStatic, PersistDynamic };
    enum class MathPrecision { Half, Float, Double };
    enum class MathType { Default, Tensorcore, TensorcoreConv };
}

template <typename T_DATA, typename T_NN> class cudaRNN::RNN
{
    private:
        cudnnHandle_t cudnnHandle;

        // for cuBLAS (dense layer)
        cublasHandle_t cublasHandle;
        T_NN alpha = 1.0;
        T_NN beta = 0.0;

        // helper structs
        RNNOptions_t options;
        optimizerOptions_t optimOpts;
        details::spRNNOptions_t spOptions;
        details::Descriptors_t descriptors;
        details::Tensors_t<T_NN> tensors;

        // tensor dimensions
        size_t inTensorSz;
        size_t hidTensorSz;
        size_t hidOutTensorSz;
        size_t tmpHidOutTensorSz;
        size_t outTensorSz;
        size_t RNNWgtTensorSz;
        size_t denseWgtTensorSz;
        int dimHidden[3], strideHidden[3];

        // non-def optimizer extra space
        T_NN *vDense, *vRNN;   // RMSProp y Adam (inicializar con 0)
        T_NN *mDense, *mRNN;   // Adam (inicializar con 0)

        // aux params
        int *seqLengthArray;
        int *devSeqLengthArray;
        unsigned long long wgtSeed = 1234ULL;

        void *workSpace;
        void *reserveSpace;
        void *states;

        size_t wgtSpaceByteSize;
        size_t workSpaceSize;
        size_t reserveSpaceSize;

        T_DATA *devInputs; // [inLength, miniBatchSz*nMiniBatches, inVecSize]
        T_DATA *devTargets; // [outLength, miniBatchSz*nMiniBatches, outVecSize]

        int nSequences;
        int nMiniBatches;

        // loss calc params
        MetricType metricType = MetricType::None;
        int nIterToCalcLoss = 1;
        std::string lossFilename;

        // mem management functions
        void allocTensors();
        void setDescriptors();
        void freeTensors();
        void destroyDescriptors();

        // extra helper functions
        void setSpOptions();
        void checkOptionErr();
        void cloneDevArr(T_NN*, int, int);
        void setMiniBatchInputs(int*, int);
        void translateLayer();
        void calc_dout(int*, int);
        void calc_dy();
        void calc_dwDense();
        float calc_loss();
        void printLoss(int, float);
        void initGPUExampleData(T_NN*, int, T_NN);

        // optimizer functions
        float b1t = 1, b2t = 1; // for Adam
               
        void updateWgtsRNN_SGD();
        void updateWgtsDense_SGD();
        void updateWgtsRNN_RMSProp();
        void updateWgtsDense_RMSProp();
        void updateWgtsRNN_Adam();
        void updateWgtsDense_Adam();

    public:
        RNN(RNNOptions_t);
        ~RNN();

        RNN(const RNN &) = delete; // copy constructor
        RNN& operator=(const RNN &) = delete; // copy assignment
        RNN(RNN &&) = delete; // move constructor
        RNN& operator=(RNN &&) = delete; // move assignment

        void setOptions(RNNOptions_t);
        void setInputs(std::vector<T_DATA>&); // [inLength, miniBatchSz*nMiniBatches, inVecSize] (en host) -> [seqLength, miniBatchSz, inVecSize] (x en device)
        void setTargets(std::vector<T_DATA>&); // [outLength, miniBatchSz*nMiniBatches, outVecSize] (en host) -> [outLength, miniBatchSz, outVecSize] (matchear out)
        void setSeed(unsigned long long);
        void setMetrics(MetricType, int, std::string); // default = don't calculate loss

        // for training
        void setOptimizer(optimizerOptions_t); // default = SGD
        void train(int epochs = 1, bool shuffleSequences = true, bool showProgress = false);

        // when training is done
        void test(int epochs = 1, bool printOutputs = false, std::string outputsFilename = "outputs.txt");

        void save(std::string wgtsFilename = "Weights.txt", std::string optsFilename = "Options.txt");
        void load(std::string wgtsFilename = "Weights.txt", std::string optsFilename = "Options.txt");

        // debugging (copied from the cuDNN example)
        void initDataExample();
        void performChecksums();
        void RNNSinglePassExample();
};


struct cudaRNN::RNNOptions_t 
{
    unsigned int seqLength       = 20; 
    unsigned int inLength        = 20; 
    unsigned int outLength       = 20; 
    unsigned int numLayers       = 2;    // Specify number of layers
    unsigned int inVecSize       = 512;  // Specify input vector size
    unsigned int outVecSize      = 512;  // Specify output vector size
    unsigned int hiddenSize      = 512;  // Specify hidden size
    unsigned int miniBatchSz     = 64;   // Specify max miniBatch size
    float        dropout         = 0;

    InputMode inputMode         = InputMode::Linear;    // Specify how the input to the RNN model is processed by the first layer (skip or linear input)
    CellMode cellMode           = CellMode::LSTM;       // Specify cell type (RELU, TANH, LSTM, GRU)
    BiasMode biasMode           = BiasMode::Double;     // Specify bias type (no bias, single inp bias, single rec bias, double bias)
    Algorithm algorithm         = Algorithm::Standard;  // Specify recurrence algorithm (standard, persist dynamic, persist static)
    MathPrecision mathPrecision = MathPrecision::Float; // Specify math precision (half, float or double)
    MathType mathType           = MathType::Default;    // Specify math type (default, tensor op math or tensor op math with conversion)    

    friend std::ostream& operator<<(std::ostream& os, const cudaRNN::RNNOptions_t& options)
    {
        os << options.seqLength          << std::endl
           << options.inLength           << std::endl
           << options.outLength          << std::endl
           << options.numLayers          << std::endl
           << options.inVecSize          << std::endl
           << options.outVecSize         << std::endl
           << options.hiddenSize         << std::endl
           << options.miniBatchSz        << std::endl
           << options.dropout            << std::endl
           << (int)options.inputMode     << std::endl
           << (int)options.cellMode      << std::endl
           << (int)options.biasMode      << std::endl
           << (int)options.algorithm     << std::endl
           << (int)options.mathPrecision << std::endl
           << (int)options.mathType;
    
        return os;
    }

    friend std::istream& operator>>(std::istream& is, cudaRNN::RNNOptions_t& options)
    {
        is >> options.seqLength
           >> options.inLength
           >> options.outLength
           >> options.numLayers
           >> options.inVecSize
           >> options.outVecSize
           >> options.hiddenSize
           >> options.miniBatchSz
           >> options.dropout;

        unsigned int tmp_inputMode = 0;
        unsigned int tmp_cellMode = 0;
        unsigned int tmp_biasMode = 0;
        unsigned int tmp_algorithm = 0;
        unsigned int tmp_mathPrecision = 0;
        unsigned int tmp_mathType = 0;
        
        is  >> tmp_inputMode
            >> tmp_cellMode
            >> tmp_biasMode
            >> tmp_algorithm
            >> tmp_mathPrecision
            >> tmp_mathType;

        options.inputMode     = static_cast<InputMode>(tmp_inputMode);
        options.cellMode      = static_cast<CellMode>(tmp_cellMode);
        options.biasMode      = static_cast<BiasMode>(tmp_biasMode);
        options.algorithm     = static_cast<Algorithm>(tmp_algorithm);
        options.mathPrecision = static_cast<MathPrecision>(tmp_mathPrecision);
        options.mathType      = static_cast<MathType>(tmp_mathType);
    
        return is;
    }
};

struct cudaRNN::optimizerOptions_t
{
    Optimizer optimizer = cudaRNN::Optimizer::SGD;

    float lr      = 0.001;  // SGD, RMSProp, Adam (learning rate)
    float gamma   = 0.9;    // RMSProp
    float b1      = 0.9;    // Adam
    float b2      = 0.999;  // Adam
    float epsilon = 1e-8;   // Adam
};

struct cudaRNN::details::spRNNOptions_t
{
    // cuDNN
    cudnnDataType_t     dataType;
    cudnnRNNInputMode_t inputMode;     // Specify how the input to the RNN model is processed by the first layer (skip or linear input)
    cudnnRNNMode_t      cellMode;      // Specify cell type (RELU, TANH, LSTM, GRU)
    cudnnRNNBiasMode_t  biasMode;      // Specify bias type (no bias, single inp bias, single rec bias, double bias)
    cudnnRNNAlgo_t      algorithm;     // Specify recurrence algorithm (standard, persist dynamic, persist static)
    cudnnDataType_t     mathPrecision; // Specify math precision (half, float of double)
    cudnnMathType_t     mathType;      // Specify math type (default, tensor op math or tensor op math with conversion)
    
    // cuBLAS
    cudaDataType_t      cuBLASDataType;
    cublasGemmAlgo_t    cuBLASMathType;
};

struct cudaRNN::details::Descriptors_t
{
    cudnnRNNDataDescriptor_t x;
    cudnnRNNDataDescriptor_t y;
    cudnnTensorDescriptor_t  h;
    cudnnTensorDescriptor_t  c;
    cudnnTensorDescriptor_t  w;
    cudnnTensorDescriptor_t  b;
    cudnnRNNDescriptor_t     rnn;
    cudnnDropoutDescriptor_t dropout;
};

template <typename T_NN> struct cudaRNN::details::Tensors_t
{
    // cuDNN
    T_NN *x, *hx, *cx;
    T_NN *dx, *dhx, *dcx;

    T_NN *y, *hy, *cy;
    T_NN *dy, *dhy, *dcy;

    T_NN *w; // has both weights and biases
    T_NN *dw;

    // cuBLAS (dense layer)
    T_NN *y_temp; // for adapting y (cuDNN) -> y_temp (cuBLAS) (row-major -> column-major)
    T_NN *dy_temp; // for adapting dy_temp (cuBLAS) -> dy (cuDNN) (column-major -> row-major)

    T_NN *out;
    T_NN *dout;

    T_NN *wDense; // without bias
    T_NN *dwDense;
};