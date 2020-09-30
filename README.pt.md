![logo titulo](https://user-images.githubusercontent.com/24854541/94681693-435a4100-02fa-11eb-85df-5cbd55374f98.png)

![divisoria](https://user-images.githubusercontent.com/24854541/94681772-5a009800-02fa-11eb-8ff8-29f9fad9b18f.png)

![subtitulo prt](https://user-images.githubusercontent.com/24854541/94694582-60e3d680-030b-11eb-9092-f316556ce91e.png)

*Read this README in [`english`](https://github.com/viniciusvviterbo/Multilayer-Perceptron/blob/master/README.md)*

A multilayer perceptron (MLP) consists of an artificial neural network with at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.

In this repository there is a parallel implementation of an MLP

# A Arquitetura MLP

(Explicação da arquitetura MLP precisa set feita)

# O Dataset

(Explicação do Dataset precisa set feita)

# Uso

Para usar o código, é necessário primeiramente clonar esse repositório.

```shell
git clone github.com/viniciusvviterbo/Multilayer-Perceptron
cd ./Multilayer-Perceptron
```

### Formatando o dataset

Nesse projeto optamos por descrever as principais características na primeira linha, uma linha vazia - para facilidade de leitura, é inteiramente opcional -, e os dados em si. Exemplo:

```
[NÚMERO DE CASOS] [NÚMERO DE ENTRADAS] [NÚMERO DE SAÍDAS]

[ENTRADA 1] [ENTRADA 2] ... [ENTRADA N] [SAÍDA 1] [SAÍDA 2] ... [SAÍDA N]
[ENTRADA 1] [ENTRADA 2] ... [ENTRADA N] [SAÍDA 1] [SAÍDA 2] ... [SAÍDA N]
[ENTRADA 1] [ENTRADA 2] ... [ENTRADA N] [SAÍDA 1] [SAÍDA 2] ... [SAÍDA N]
```

Para testar o código, nesse repositório está incluso o dataset para a porta lógica XOR ([pattern_logic-port.in](https://github.com/viniciusvviterbo/Multilayer-Perceptron/blob/master/pattern_logic-port.in)), e ele pode ser usado para melhor entender a formatação necessária.

### Normalizando o dataset

Um dataset normalizado é preferível por seus resultados (quase) absolutos dados ao fim do treinamento: 0 ou 1. Para normalizar o dataset, execute:

```shell
g++ ./normalizeDataset.cpp -o ./normalizeDataset
./normalizeDataset.cpp < PATTERN_FILE > NORMALIZED_PATTERN_FILE
```

Exemplo:

```shell
g++ ./normalizeDataset.cpp -o ./normalizeDataset
./normalizeDataset.cpp < pattern_breast-cancer.in > normalized_pattern_breast-cancer.in
```

### Compilando o código fonte

Compile o código usando OpenMP

```shell
g++ ./mlp.cpp -o ./mlp -fopenmp
```

### Treinamento e Resultado

Nesse código, nós dividimos o dataset informado pela metade. A primeira metade é usada apenas para treinamento, a segunda é usada para teste, desse modo a rede vê essa segunda parte como novo conteúdo e tenta obter o diagnóstico correto.

Os resultados esperados e os obtidos pela MLP são impressos para comparação.

### Executando

Para execução, o comando necessita alguns parâmetros:

```shell
.mlp TAMANHO_CAMADA_ESCONDIDA TAXA_DE_APRENDIZADO LIMIAR < ARQUIVO_DE_PADRÕES
```

* TAMANHO_CAMADA_ESCONDIDA  se refere ao número de neurônios na camada escondida da rede;
* TAXA_DE_APRENDIZADO se refere à taxa de aprendizado da rede, um número de ponto flutuante usado durante a fase de correção do backpropagation;
* LIMIAR  se refere à maior taxa de erro aceita pela rede a fim de obter um resultado correto;
* ARQUIVO_DE_PADRÕES refers to the normalized pattern file

Exemplo:

```shell
.mlp 5 0.2 1e-5 < ./normalized_pattern_breast-cancer.in
```

# Referências

[Canal do Youtube de Fabrício Goés](https://www.youtube.com/channel/UCgeFcHndjZVth6HRg3cFkng) - por [Dr. Luis Goés](http://lattes.cnpq.br/7401444661491250)

[Eitas Tutoriais](http://www.eitas.com.br/tutoriais/12) - por [Espaço de Inovação Tecnológica Aplicada e Social - PUC Minas](http://www.eitas.com.br/)

[Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) - de [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

[Koliko](http://www.fontslots.com/koliko-font/) - por [Alex Frukta](https://www.behance.net/MRfrukta) & [Vladimir Tomin](https://www.behance.net/myaka)


![divisoria](https://user-images.githubusercontent.com/24854541/94681772-5a009800-02fa-11eb-8ff8-29f9fad9b18f.png)

**[GNU AGPL v3.0](https://www.gnu.org/licenses/agpl-3.0.html)**
