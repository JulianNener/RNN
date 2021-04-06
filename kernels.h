#pragma once

#include "fp16_emu.h"

namespace cudaRNN
{   

namespace details

{

template <typename T_NN>
__global__ void cloneDevArr_ker(T_NN *arr, int arrSz, int nClones)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < arrSz*(nClones-1))
    {
        int arr_idx = id/arrSz;
        int elem_idx = id%arrSz;

        arr[idx_2Dto1D(arr_idx+1, elem_idx, arrSz)] = arr[idx_2Dto1D(0, elem_idx, arrSz)];
    }
}

template <typename T_NN>
__global__ void translateLayer_ker(T_NN *y, T_NN *y_temp, int seqLength, int outLength, int miniBatchSz, int hiddenSize, int nElems)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < nElems)
    {
        int seqL_idx = id/(miniBatchSz*hiddenSize); // = t_idx
        int temp_idx = id%(miniBatchSz*hiddenSize);

        int seq_idx = temp_idx/hiddenSize;
        int hid_idx = temp_idx%hiddenSize;

        int seqL_offset = seqLength - outLength;

        int y_idx = idx_3Dto1D(seqL_idx, seq_idx, hid_idx, miniBatchSz, hiddenSize); // row-major
        int y_temp_idx = idx_3Dto1D(hid_idx, seq_idx, seqL_idx - seqL_offset, miniBatchSz, outLength); // column-major

        if(seqL_idx >= seqL_offset) y_temp[y_temp_idx] = y[y_idx];
    }
}

template <typename T_NN>
__global__ void calc_dy_ker(T_NN *dy, T_NN *dy_temp, int hiddenSize, int seqLength, int outLength, int miniBatchSz, int nElems)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < nElems)
    {  
        int seqL_idx = id/(miniBatchSz*hiddenSize); // = t_idx
        int temp_idx = id%(miniBatchSz*hiddenSize);

        int seq_idx = temp_idx/hiddenSize;
        int hid_idx = temp_idx%hiddenSize;

        int seqL_offset = seqLength - outLength;

        int dy_idx = idx_3Dto1D(seqL_idx, seq_idx, hid_idx, miniBatchSz, hiddenSize); // row-major
        int dy_temp_idx = idx_3Dto1D(hid_idx, seq_idx, seqL_idx - seqL_offset, miniBatchSz, outLength); // column-major

        dy[dy_idx] = (seqL_idx < seqL_offset) ? 0.0 : dy_temp[dy_temp_idx];
    }
}

template <typename T_NN>
__global__ void calc_dwDense_ker(T_NN *dwDense, int miniBatchSz, int outLength, int hiddenSize, int nWgts)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < nWgts)
    {
        int nWgtMats = miniBatchSz * outLength;

        T_NN dw_avg = 0;
        for(int i = 0; i < nWgtMats; ++i)
            dw_avg += dwDense[idx_2Dto1D(i, id, nWgts)];       
            
        dw_avg /= nWgtMats;

        for(int i = 0; i < nWgtMats; ++i)
            dwDense[idx_2Dto1D(i, id, nWgts)] = dw_avg;
    }
}

template <typename T_NN>
__global__ void updateWgtsRNN_SGD_ker(T_NN* w, T_NN* dw, float lr, int RNNWgtTensorSz)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < RNNWgtTensorSz)
        w[id] -= static_cast<T_NN>(lr)*dw[id]; 
}


template <typename T_NN>
__global__ void updateWgtsDense_SGD_ker(T_NN *wDense, T_NN *dwDense, float lr, int denseWgtTensorSz)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < denseWgtTensorSz)
        wDense[id] -= static_cast<T_NN>(lr)*dwDense[id];
}

template <typename T_NN>
__global__ void updateWgtsRNN_RMSProp_ker(T_NN* w, T_NN* dw, T_NN* v, float lr, float gamma, int RNNWgtTensorSz)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < RNNWgtTensorSz)
    {
        T_NN my_dw = dw[id];
        T_NN my_v = gamma*v[id] + (1 - gamma)*my_dw*my_dw;

        w[id] -= (lr/sqrtf(my_v)) * my_dw;
        v[id] = my_v; 
    }
}

template <>
__global__ void updateWgtsRNN_RMSProp_ker<half1>(half1* w, half1* dw, half1* v, float lr, float gamma, int RNNWgtTensorSz)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < RNNWgtTensorSz)
    {
        half1 my_dw = dw[id];
        half1 hgamma = __float2half(gamma);

        half1 my_v = hgamma*v[id] + (__int2half_rn(1) - hgamma)*my_dw*my_dw;

        w[id] -= __float2half( (lr/(sqrtf(my_v)+HLF_MIN)) ) * my_dw;
        v[id] = my_v; 
    }
}

template <typename T_NN>
__global__ void updateWgtsDense_RMSProp_ker(T_NN* wDense, T_NN* dwDense, T_NN* v, float lr, float gamma, int denseWgtTensorSz, int nWgts)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < denseWgtTensorSz)
    {
        int mat_idx = id/nWgts;
        int w_idx = id%nWgts;
        int my_idx = idx_2Dto1D(mat_idx, w_idx, nWgts);

        T_NN my_dw = dwDense[my_idx];
        T_NN my_v = gamma*v[w_idx] + (1 - gamma)*my_dw*my_dw;

        wDense[my_idx] -= (lr/sqrtf(my_v)) * my_dw;

        if(mat_idx == 0) v[w_idx] = my_v; // v has the size of a single wgt matrix -> avoid collisions!
    }
}

template <>
__global__ void updateWgtsDense_RMSProp_ker<half1>(half1* wDense, half1* dwDense, half1* v, float lr, float gamma, int denseWgtTensorSz, int nWgts)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < denseWgtTensorSz)
    {
        int mat_idx = id/nWgts;
        int w_idx = id%nWgts;
        int my_idx = idx_2Dto1D(mat_idx, w_idx, nWgts);
        half1 hgamma = __float2half(gamma);

        half1 my_dw = dwDense[my_idx];
        half1 my_v = hgamma*v[w_idx] + (__int2half_rn(1) - hgamma)*my_dw*my_dw;

        wDense[my_idx] -= __float2half( (lr/(sqrtf(my_v) + HLF_MIN)) ) * my_dw;

        if(mat_idx == 0) v[w_idx] = my_v; // v has the size of a single wgt matrix -> avoid collisions!
    }
}

template <typename T_NN>
__global__ void updateWgtsRNN_Adam_ker(T_NN* w, T_NN* dw, T_NN* m, T_NN* v, 
                                       float lr, float b1, float b2, float b1t, float b2t, float epsilon, 
                                       int RNNWgtTensorSz)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < RNNWgtTensorSz)
    {
        T_NN my_m = m[id], my_v = v[id];
        T_NN my_dw = dw[id];

        my_m = b1*my_m + (1 - b1)*my_dw;
        my_v = b2*my_v + (1 - b2)*my_dw*my_dw;

        T_NN my_m_hat = my_m/(1 - b1t);
        T_NN my_v_hat = my_v/(1 - b2t);

        w[id] -= lr * ( my_m_hat/(sqrtf(my_v_hat) + epsilon) );     

        m[id] = my_m;
        v[id] = my_v;
    }
}

template <>
__global__ void updateWgtsRNN_Adam_ker<half1>(half1* w, half1* dw, half1* m, half1* v, 
                                              float lr, float b1, float b2, float b1t, float b2t, float epsilon, 
                                              int RNNWgtTensorSz)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < RNNWgtTensorSz)
    {
        half1 my_m = m[id], my_v = v[id];
        half1 my_dw = dw[id];

        half1 hb1 = __float2half(b1);
        half1 hb2 = __float2half(b2);
        half1 hb1t = __float2half(b1t);
        half1 hb2t = __float2half(b2t);

        my_m = hb1*my_m + (__int2half_rn(1) - hb1)*my_dw;
        my_v = hb2*my_v + (__int2half_rn(1) - hb2)*my_dw*my_dw;

        half1 my_m_hat = my_m/(__int2half_rn(1) - hb1t);
        half1 my_v_hat = my_v/(__int2half_rn(1) - hb2t);

        w[id] -= __float2half(lr * ( __half2float(my_m_hat)/(sqrtf(my_v_hat) + HLF_MIN + epsilon ) ) );     

        m[id] = my_m;
        v[id] = my_v;
    }
}

template <typename T_NN>
__global__ void updateWgtsDense_Adam_ker(T_NN* w, T_NN* dw, T_NN* m, T_NN* v, 
                                         float lr, float b1, float b2, float b1t, float b2t, float epsilon, 
                                         int denseWgtTensorSz, int nWgts)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < denseWgtTensorSz)
    {
        int mat_idx = id/nWgts;
        int w_idx = id%nWgts;
        int my_idx = idx_2Dto1D(mat_idx, w_idx, nWgts);

        T_NN my_m = m[w_idx], my_v = v[w_idx];
        T_NN my_dw = dw[my_idx];

        my_m = b1*my_m + (1 - b1)*my_dw;
        my_v = b2*my_v + (1 - b2)*my_dw*my_dw;

        T_NN my_m_hat = my_m/(1 - b1t);
        T_NN my_v_hat = my_v/(1 - b2t);

        w[my_idx] -= lr * ( my_m_hat/(sqrtf(my_v_hat) + epsilon) );

        if(mat_idx == 0)
        {
            m[w_idx] = my_m;
            v[w_idx] = my_v;
        }        
    }
}

template <>
__global__ void updateWgtsDense_Adam_ker<half1>(half1* w, half1* dw, half1* m, half1* v, 
                                                float lr, float b1, float b2, float b1t, float b2t, float epsilon, 
                                                int denseWgtTensorSz, int nWgts)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < denseWgtTensorSz)
    {
        int mat_idx = id/nWgts;
        int w_idx = id%nWgts;
        int my_idx = idx_2Dto1D(mat_idx, w_idx, nWgts);

        half1 my_m = m[w_idx], my_v = v[w_idx];
        half1 my_dw = dw[my_idx];

        half1 hb1 = __float2half(b1);
        half1 hb2 = __float2half(b2);
        half1 hb1t = __float2half(b1t);
        half1 hb2t = __float2half(b2t);

        my_m = hb1*my_m + (__int2half_rn(1) - hb1)*my_dw;
        my_v = hb2*my_v + (__int2half_rn(1) - hb2)*my_dw*my_dw;

        half1 my_m_hat = my_m/(__int2half_rn(1) - hb1t);
        half1 my_v_hat = my_v/(__int2half_rn(1) - hb2t);

        w[my_idx] -= __float2half(lr * ( __half2float(my_m_hat)/(sqrtf(my_v_hat) + HLF_MIN + epsilon ) ) );

        if(mat_idx == 0)
        {
            m[w_idx] = my_m;
            v[w_idx] = my_v;
        }        
    }
}

template <typename T_NN>
__global__ void squareElems_ker(T_NN *arr, int nElems)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < nElems)
        arr[id] *= arr[id];
}

template <typename T_NN>
__global__ void absValElems_ker(T_NN *arr, int nElems)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < nElems)
        arr[id] = fabsf(arr[id]);
}

template <typename T_NN> 
__global__ void initGPUExampleData_ker(T_NN *data, int numElements, T_NN value) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements) {
        data[tid] = value;
    }
}

template <typename T_DATA, typename T_NN> 
__global__ void setMiniBatchInputs_ker(T_DATA *inputs, T_NN *x, int inTensorSz, int nSequences, 
                                       int inLength, int seqLength, int miniBatchSz, int inVecSize, int *shuffledSeqIdx, int miniB_idx)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < inTensorSz)
    {
        int seqL_idx = id/(miniBatchSz*inVecSize); // = t_idx
        int temp_idx = id%(miniBatchSz*inVecSize);

        int seq_idx = temp_idx/inVecSize;
        int in_idx = temp_idx%inVecSize;

        int randSeq_idx = shuffledSeqIdx[miniB_idx*miniBatchSz + seq_idx];

        int in_elem_idx = idx_3Dto1D(seqL_idx, randSeq_idx, in_idx, nSequences, inVecSize);
        int x_elem_idx = idx_3Dto1D(seqL_idx, seq_idx, in_idx, miniBatchSz, inVecSize);

        x[x_elem_idx] = (seqL_idx < inLength) ? inputs[in_elem_idx] : 0;
    }
}

template <typename T_DATA, typename T_NN> 
__global__ void calc_dout_ker(T_NN *out, T_NN *dout, T_DATA *targets, 
                              int miniBatchSz, int outLength, int outVecSize, int *shuffledSeqIdx, int miniB_idx, int nSequences, int nElems)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < nElems)
    {
        int seqL_idx = id/(miniBatchSz*outVecSize); // = t_idx
        int temp_idx = id%(miniBatchSz*outVecSize);

        int seq_idx = temp_idx/outVecSize;
        int out_idx = temp_idx%outVecSize;  

        int randSeq_idx = shuffledSeqIdx[miniB_idx*miniBatchSz + seq_idx];

        int elem_idx = idx_3Dto1D(out_idx, seq_idx, seqL_idx, miniBatchSz, outLength); // column-major (comes from cuBLAS)
        int tgt_idx = idx_3Dto1D(seqL_idx, randSeq_idx, out_idx, nSequences, outVecSize); // row-major (like devInputs, convention)

        dout[elem_idx] = out[elem_idx] - static_cast<T_NN>(targets[tgt_idx]);
    }
}

__global__ void float2halfArr_ker(float* fArr, half1* halfArr, int nElems)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < nElems) {
        halfArr[id] = __float2half(fArr[id]);
    }
}

} // namespace details
} // namespace cudaRNN