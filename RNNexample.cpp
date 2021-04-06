#include "RNN.h"
#include <cmath>

using namespace std;
using namespace cudaRNN;

template <typename T>
vector<T> makeInputData(RNNOptions_t myOpts, int nSeq)
{
    vector<T> myInputs(myOpts.inLength * nSeq * myOpts.inVecSize); // [inLength, nSeq = miniBatchSz*nMiniBatches, inVecSize]

    for(int i = 0; i < myOpts.inLength; ++i)
    {
        for(int j = 0; j < nSeq; ++j)
        {
            for(int k = 0; k < myOpts.inVecSize; ++k)
                myInputs[details::idx_3Dto1D(i, j, k, nSeq, myOpts.inVecSize)] = sinf(1.*i/(j+1));
        }
    }

    ofstream out("pyinputs.txt");
    for(int i = 0; i < nSeq; ++i)
    {
        for(int j = 0; j < myOpts.inLength; ++j)
        {
            for(int k = 0; k < myOpts.inVecSize; ++k)
                out << myInputs[details::idx_3Dto1D(j, i, k, nSeq, myOpts.outVecSize)] << " ";
            out << endl;
        }
        out << endl << endl;
    }

    return myInputs;
}

template <typename T>
vector<T> makeTargetData(RNNOptions_t myOpts, int nSeq)
{
    vector<T> myTargets(myOpts.outLength * nSeq * myOpts.outVecSize); // [outLength, nSeq = miniBatchSz*nMiniBatches, outVecSize]

    for(int i = 0; i < myOpts.outLength; ++i)
    {
        for(int j = 0; j < nSeq; ++j)
        {
            for(int k = 0; k < myOpts.outVecSize; ++k)
                myTargets[details::idx_3Dto1D(i, j, k, nSeq, myOpts.outVecSize)] = sinf(1.*(i+k+myOpts.inLength)/(j+1));
        }
    }

    ofstream pyout("pytargets.txt");
    for(int i = 0; i < nSeq; ++i)
    {
        for(int j = 0; j < myOpts.outLength; ++j)
        {
            for(int k = 0; k < myOpts.outVecSize; ++k)
                pyout << myTargets[details::idx_3Dto1D(j, i, k, nSeq, myOpts.outVecSize)] << " ";
            pyout << endl;
        }
        //out << endl << endl;
    }

    ofstream out("targets.txt");
    for(int i = 0; i < nSeq; ++i)
    {
        for(int j = 0; j < myOpts.outLength; ++j)
        {
            for(int k = 0; k < myOpts.outVecSize; ++k)
                out << myTargets[details::idx_3Dto1D(j, i, k, nSeq, myOpts.outVecSize)] << " ";
            out << endl;
        }
        out << endl << endl;
    }

    return myTargets;
}


int main()
{
    RNNOptions_t myOpts;
    myOpts.numLayers = 2;
    myOpts.seqLength = 40;
    myOpts.inLength = 30;
    myOpts.outLength = 10;
    myOpts.miniBatchSz = 1;
    myOpts.inVecSize = 1;
    myOpts.outVecSize = 1;
    myOpts.hiddenSize = 32; 
    myOpts.cellMode = CellMode::LSTM;
    myOpts.mathPrecision = MathPrecision::Float;
    myOpts.algorithm = Algorithm::PersistStatic;
    myOpts.biasMode = BiasMode::Double;
    myOpts.mathType = MathType::Default;

    /*** To compare with the CUDA sample ***/  
    // RNN<float, float> myRNN(myOpts);
    // myRNN.initDataExample();
    // myRNN.RNNSinglePassExample();
    // myRNN.performChecksums();
    /***************************************/

    int nSeq = 10;
    vector<float> myInputs = makeInputData<float>(myOpts, nSeq);
    vector<float> myTargets = makeTargetData<float>(myOpts, nSeq);

    RNN<float, float> myRNN(myOpts);
    myRNN.setInputs(myInputs);
    myRNN.setTargets(myTargets);

    optimizerOptions_t myOptim;
    myOptim.optimizer = Optimizer::Adam;

    myRNN.setOptimizer(myOptim);
    myRNN.setMetrics(MetricType::MSE, 1, "loss.txt");
    myRNN.train(1000, true, true);

    myRNN.setMetrics(MetricType::None, 0, "");
    myRNN.test(1, true, "outputs.txt");
    myRNN.save();


    // RNNOptions_t myOpts2;
    // RNN<float, float> myRNN2(myOpts2);
    // myRNN2.load();

    // myRNN2.setInputs(myInputs);
    // myRNN2.setTargets(myTargets);
    // myRNN2.test(1, true, "outputs.txt");

    return 0;
}