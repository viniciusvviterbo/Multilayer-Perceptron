![logo titulo](https://user-images.githubusercontent.com/24854541/94681693-435a4100-02fa-11eb-85df-5cbd55374f98.png)

![divisoria](https://user-images.githubusercontent.com/24854541/94681772-5a009800-02fa-11eb-8ff8-29f9fad9b18f.png)

![subtitulo eng](https://user-images.githubusercontent.com/24854541/94682006-afd54000-02fa-11eb-8657-265172fbc4f8.png)

*Leia esse README em [`português`](https://github.com/viniciusvviterbo/Multilayer-Perceptron/blob/master/README.pt.md).*

A multilayer perceptron (MLP) consists of an artificial neural network with at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

In this repository there is a parallel implementation of an MLP

# The MLP Architecture

(MLP architecture explanation to be done)

# The Dataset

(Dataset explanation to be done)

# Usage

In order to use the code, you need to first and foremost clone this repository.

```shell
git clone github.com/viniciusvviterbo/Multilayer-Perceptron
cd ./Multilayer-Perceptron
```

### Formatting the dataset

In this project we opted for describing the main informations in the first line, an empty line - for ease of read, it is entirely optional -, and the data itself. Example:

```
[NUMBER OF CASES] [NUMBER OF INPUTS] [NUMBER OF OUTPUTS]

[INPUT 1] [INPUT 2] ... [INPUT N] [OUTPUT 1] [OUTPUT 2] ... [OUTPUT N]
[INPUT 1] [INPUT 2] ... [INPUT N] [OUTPUT 1] [OUTPUT 2] ... [OUTPUT N]
[INPUT 1] [INPUT 2] ... [INPUT N] [OUTPUT 1] [OUTPUT 2] ... [OUTPUT N]
```

For testing the code, in this repository we included the dataset for the XOR logical port ([pattern_logic-port.in](https://github.com/viniciusvviterbo/Multilayer-Perceptron/blob/master/pattern_logic-port.in)), and it can be used for better understanding the needed format.

### Normalizing the dataset

A normalized dataset is preferred for its (kinda) absolute results given at the end of training: 0 or 1. To normalize the dataset, execute:

```shell
g++ ./normalizeDataset.cpp -o ./normalizeDataset
./normalizeDataset.cpp < PATTERN_FILE > NORMALIZED_PATTERN_FILE
```

Example:

```shell
g++ ./normalizeDataset.cpp -o ./normalizeDataset
./normalizeDataset.cpp < pattern_breast-cancer.in > normalized_pattern_breast-cancer.in
```

### Compiling the source code

Compile the source code using OpenMP

```shell
g++ ./mlp.cpp -o ./mlp -fopenmp
```

### Training and Result

In this code, we are dividing the dataset informed by half. The first half is used for training purposes only, the second one is used for testing, this way the network sees the latter half as new content and tries to obtain the correct result.

The expected results and the ones obtained by the MLP are printed for comparison.

### Executing

For executing, the command needs some parameters:

```shell
.mlp HIDDEN_LAYER_LENGTH TRAINING_RATE THRESHOLD < PATTERN_FILE
```

* HIDDEN_LAYER_LENGTH refers to the number of neurons in the network hidden layer;
* TRAINING_RATE refers to the network's rate of training, a floating point number used during the correction phase of backpropagation;
* THRESHOLD  refers to the maximum error admitted by the network in order to obtain an acceptably correct result;
* PATTERN_FILE refers to the normalized pattern file

Example:

```shell
.mlp 5 0.2 1e-5 < ./normalized_pattern_breast-cancer.in
```

# References

[Fabrício Goés Youtube Channel](https://www.youtube.com/channel/UCgeFcHndjZVth6HRg3cFkng) - by [Dr. Luis Goés](http://lattes.cnpq.br/7401444661491250)

[Eitas Tutoriais](http://www.eitas.com.br/tutoriais/12) - by [Espaço de Inovação Tecnológica Aplicada e Social - PUC Minas](http://www.eitas.com.br/)

[Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) - from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

[Koliko](http://www.fontslots.com/koliko-font/) - by [Alex Frukta](https://www.behance.net/MRfrukta) & [Vladimir Tomin](https://www.behance.net/myaka)


![divisoria](https://user-images.githubusercontent.com/24854541/94681772-5a009800-02fa-11eb-8ff8-29f9fad9b18f.png)

**[GNU AGPL v3.0](https://www.gnu.org/licenses/agpl-3.0.html)**
