![logo titulo](https://user-images.githubusercontent.com/24854541/94681693-435a4100-02fa-11eb-85df-5cbd55374f98.png)

![divisoria](https://user-images.githubusercontent.com/24854541/94681772-5a009800-02fa-11eb-8ff8-29f9fad9b18f.png)

![subtitulo eng](https://user-images.githubusercontent.com/24854541/94682006-afd54000-02fa-11eb-8657-265172fbc4f8.png)

*Leia esse README em [`português`](https://github.com/viniciusvviterbo/Multilayer-Perceptron/blob/master/README.pt.md).*

A multilayer perceptron (MLP) consists of an **Artificial Neural Network** with at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

In this repository there is a parallel implementation of an MLP that recognizes characters regardless of the font it is written in.

# The Dataset

The original dataset consists of images from 153 character fonts obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Character+Font+Images). Some fonts were scanned from a variety of devices: hand scanners, desktop scanners or cameras. Other fonts were computer generated.

# Usage

In order to use the code, you need to first and foremost clone this repository.

```shell
git clone github.com/viniciusvviterbo/Multilayer-Perceptron
cd ./Multilayer-Perceptron
```

### Formatting the Dataset

In this project we opted for describing the main informations in the first line, an empty line - for ease of read, it is entirely optional -, and the data itself. Example:

```
[NUMBER OF CASES] [NUMBER OF INPUTS] [NUMBER OF OUTPUTS]

[INPUT 1] [INPUT 2] ... [INPUT N] [OUTPUT 1] [OUTPUT 2] ... [OUTPUT N]
[INPUT 1] [INPUT 2] ... [INPUT N] [OUTPUT 1] [OUTPUT 2] ... [OUTPUT N]
[INPUT 1] [INPUT 2] ... [INPUT N] [OUTPUT 1] [OUTPUT 2] ... [OUTPUT N]
```

For testing the code, we included a reduced dataset ([sampleNormalizedFonts.in](https://github.com/viniciusvviterbo/Multilayer-Perceptron/blob/master/datasets/sampleNormalizedFonts.in)), and it can be used for better understanding the needed format.

### Normalizing the dataset

A normalized dataset is preferred for its (kind of) absolute results given at the end of training: 0 or 1. To normalize the dataset, execute:

```shell
g++ ./normalizeDataset.cpp -o ./normalizeDataset
./normalizeDataset.cpp < PATTERN_FILE > NORMALIZED_PATTERN_FILE
```

Example:

```shell
g++ ./normalizeDataset.cpp -o ./normalizeDataset
./normalizeDataset.cpp < ./datasets/patternFonts.in > ./datasets/normalizedPatternFonts.in
```

### Compiling the source code

Compile the source code using OpenMP

```shell
g++ mlp.cpp -o mlp -O3 -fopenmp -std=c++14
```

### Training and Result

In this code, we are dividing the dataset informed by half. The first half is used for training purposes only, the second one is used for testing, this way the network sees the latter half as new content and tries to obtain the correct result.

### Executing

For executing, the command needs some parameters:

```shell
./mlp HIDDEN_LAYER_LENGTH TRAINING_RATE THRESHOLD NUMBER_OF_THREADS < PATTERN_FILE
```

* HIDDEN_LAYER_LENGTH refers to the number of neurons in the network hidden layer;
* TRAINING_RATE refers to the network's rate of training, a floating point number used during the correction phase of backpropagation;
* THRESHOLD  refers to the maximum error admitted by the network in order to obtain an acceptably correct result;
* NUMBER_OF_THREADS refers to the number of threads that the network is allowed to use;
* PATTERN_FILE refers to the normalized pattern file

Example:

```shell
./mlp 1024 0.1 1e-3 4 < ./datasets/normalizedPatternFonts.in
```

As a more handy way to execute, we included in this repository a shell script to facilitate testing and seeing results from multiple executions in order to obtain an average runtime.

```shell
./script.sh
```

The script compiles the code as a sequencial implementation and runs it 5 times, then compiles it again as a parallel implementation and runs it 5 more times. For that, we are using the (already normalized and formated) reduced dataset [sampleNormalizedFonts.in](https://github.com/viniciusvviterbo/Multilayer-Perceptron/blob/master/datasets/sampleNormalizedFonts.in).

# References

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Análise do Desempenho de uma Implementação Paralela da Rede Neural Perceptron Multicamadas Utilizando Variável Compartilhada - by GÓES, Luís F. W. et al, PUC Minas 

[Introdução a Redes Neurais Multicamadas](https://www.youtube.com/watch?v=fRz57JSpl80) - by [Prof. Fagner Christian Paes](http://lattes.cnpq.br/3446751413682046)

[O que é a Multilayer Perceptron](https://www.youtube.com/watch?v=q3noPM9gcd8&list=PLKWX1jIoUZaWY_4zxjLXnIMU1Suyaa4VX&index=16) - from [ML4U](https://www.youtube.com/c/ML4U_Mello/)

[Fabrício Goés Youtube Channel](https://www.youtube.com/channel/UCgeFcHndjZVth6HRg3cFkng) - by [Dr. Luis Goés](http://lattes.cnpq.br/7401444661491250)

[Eitas Tutoriais](http://www.eitas.com.br/tutoriais/12) - by [Espaço de Inovação Tecnológica Aplicada e Social - PUC Minas](http://www.eitas.com.br/)

[Koliko](http://www.fontslots.com/koliko-font/) - by [Alex Frukta](https://www.behance.net/MRfrukta) & [Vladimir Tomin](https://www.behance.net/myaka)


![divisoria](https://user-images.githubusercontent.com/24854541/94681772-5a009800-02fa-11eb-8ff8-29f9fad9b18f.png)

**[GNU AGPL v3.0](https://www.gnu.org/licenses/agpl-3.0.html)**
