# cudaRNN
Minimal library that uses cuDNN to implement efficient RNNs in GPU

## Benchmarks
In what follows, some benchmarks are shown comparing this implementation with respect to TensorFlow using a GTX 1070 Ti GPU.

Comparison of memory used by TensorFlow with respect to this implementation, as a function of ```hiddenSize.```

![mem vs hiddenSize](https://user-images.githubusercontent.com/82059515/113897863-b230bf00-97a1-11eb-926e-9491edb9d651.png)


Speedup obtained for this implementation with respect to TensorFlow as a function of the sequence length ```seqLength```, for both LSTM and GRU cells:

![speedup vs seqLength](https://user-images.githubusercontent.com/82059515/113898067-e6a47b00-97a1-11eb-8da6-1e9be66ff87c.png)


Speedup obtained with respect to TensorFlow as a function of the number of hidden units ```hiddenSize:```

![speedup vs hiddenSize](https://user-images.githubusercontent.com/82059515/113898061-e5734e00-97a1-11eb-9845-052fe8b00b13.png)


Time per iteration in ms as a function of ```hiddenSize``` for LSTM cells. Use static persistent kernels while possible:

![time vs hiddenSize](https://user-images.githubusercontent.com/82059515/113898072-e73d1180-97a1-11eb-8265-4b0f084bc85e.png)


## How to use

The library is contained within the ```cudaRNN``` namespace. The workflow is very straightforward and similar to TensorFlow.

- Initialize the structure ```cudaRNN::RNNOptions_t```
- Instantiate ```cudaRNN::RNN``` using the previous structure. This class is templatized with 2 arguments: the first one refers to the data type of the inputs and targets (```int, float, or double```), and the second one to the data type of the weights (```__half, float or double```).
- Initialize inputs, which should be ordered as ```[inLength, nSequences, inVecSize]```, and targets as ```[outLength, nSequences, inVecSize]```, by using the methods ```setInputs``` and ```setTargets```.
- Select an optimizer and a loss metric through ```setOptimzer``` and ```setMetrics``` (optional).
- Call ```train```.
