![logo titulo](https://user-images.githubusercontent.com/24854541/94681693-435a4100-02fa-11eb-85df-5cbd55374f98.png)

![divisoria](https://user-images.githubusercontent.com/24854541/94681772-5a009800-02fa-11eb-8ff8-29f9fad9b18f.png)

![subtitulo prt](https://user-images.githubusercontent.com/24854541/94694582-60e3d680-030b-11eb-9092-f316556ce91e.png)

*Read this README in [`english`](https://github.com/viniciusvviterbo/Multilayer-Perceptron/blob/master/README.md)*

Um perceptron multicamadas (MLP) consiste em uma **Rede Neural Artificial** com pelo menos três camadas de nós: uma camada de entrada, uma camada escondida e uma camada de saída. Exceto pelos nós de entrada, cada nó é um neuron que usa uma função de ativação não linear. MLP utiliza uma técnica de aprendizado supervisionado chamado backpropagation para treinamento. Suas múltiplas camadas e ativação não-linear distingue a MLP de um perceptron linear. A rede pode distinguir dados que não são linearmente separáveis.

Nesse repositório há uma implementação paralela de uma MLP que reconhece caracteres independente da fonte em que foram escritos.

# O Dataset

O dataset original consiste de imagens de 153 fontes de caracteres obtidas do [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Character+Font+Images). Algumas fontes foram escaneadas por vários dispositivos: scanners de mão, desktop scanners ou câmeras. Outras fontes foram geradas por computador.

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

Para testar o código, nesse repositório está incluso o dataset reduzido ([sampleNormalizedFonts.in](https://github.com/viniciusvviterbo/Multilayer-Perceptron/blob/master/datasets/sampleNormalizedFonts.in)), e ele pode ser usado para melhor entender a formatação necessária.

### Normalizando o dataset

Um dataset normalizado é preferível por seus resultados (quase) absolutos dados ao fim do treinamento: 0 ou 1. Para normalizar o dataset, execute:

```shell
g++ ./normalizeDataset.cpp -o ./normalizeDataset
./normalizeDataset.cpp < PATTERN_FILE > NORMALIZED_PATTERN_FILE
```

Exemplo:

```shell
g++ ./normalizeDataset.cpp -o ./normalizeDataset
./normalizeDataset.cpp < ./datasets/patternFonts.in > ./datasets/normalizedPatternFonts.in
```

### Compilando o código fonte

Compile o código usando OpenMP

```shell
g++ mlp.cpp -o mlp -O3 -fopenmp -std=c++14
```

### Treinamento e Resultado

Nesse código, nós dividimos o dataset informado pela metade. A primeira metade é usada apenas para treinamento, a segunda é usada para teste, desse modo a rede vê essa segunda parte como novo conteúdo e tenta obter o diagnóstico correto.

### Executando

Para execução, o comando necessita alguns parâmetros:

```shell
.mlp TAMANHO_CAMADA_ESCONDIDA TAXA_DE_APRENDIZADO LIMIAR NÚMERO_DE_NÚCLEOS< ARQUIVO_DE_PADRÕES
```

* TAMANHO_CAMADA_ESCONDIDA  se refere ao número de neurônios na camada escondida da rede;
* TAXA_DE_APRENDIZADO se refere à taxa de aprendizado da rede, um número de ponto flutuante usado durante a fase de correção do backpropagation;
* LIMIAR  se refere à maior taxa de erro aceita pela rede a fim de obter um resultado correto;
* NÚMERO_DE_NÚCLEOS se refere ao número de núcleos da CPU que a rede pode utilizar;
* ARQUIVO_DE_PADRÕES refers to the normalized pattern file

Exemplo:

```shell
./mlp 1024 0.1 1e-3 4 < ./datasets/normalizedPatternFonts.in
```

Como um meio mais prático de executar, incluímos nesse repositório um script shell para facilitar os testes e ver os resultados de múltiplas execuções para obter o tempo de execução médio.

```shell
./script.sh
```

O script compila o código como uma implementação sequencial e a executa 5 vezes, depois o compila novamente como uma implementação paralela e executa mais 5 vezes. Para isso, estamos usando o (já normalizado e formatado) dataset reduzido [sampleNormalizedFonts.in](https://github.com/viniciusvviterbo/Multilayer-Perceptron/blob/master/datasets/sampleNormalizedFonts.in).

# Referências

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

Análise do Desempenho de uma Implementação Paralela da Rede Neural Perceptron Multicamadas Utilizando Variável Compartilhada - por GÓES, Luís F. W. et al, PUC Minas 

[Introdução a Redes Neurais Multicamadas](https://www.youtube.com/watch?v=fRz57JSpl80) - por [Prof. Fagner Christian Paes](http://lattes.cnpq.br/3446751413682046)

[O que é a Multilayer Perceptron](https://www.youtube.com/watch?v=q3noPM9gcd8&list=PLKWX1jIoUZaWY_4zxjLXnIMU1Suyaa4VX&index=16) - de [ML4U](https://www.youtube.com/c/ML4U_Mello/)

[Fabrício Goés Youtube Channel](https://www.youtube.com/channel/UCgeFcHndjZVth6HRg3cFkng) - por [Dr. Luis Goés](http://lattes.cnpq.br/7401444661491250)

[Eitas Tutoriais](http://www.eitas.com.br/tutoriais/12) - por [Espaço de Inovação Tecnológica Aplicada e Social - PUC Minas](http://www.eitas.com.br/)

[Koliko](http://www.fontslots.com/koliko-font/) - por [Alex Frukta](https://www.behance.net/MRfrukta) & [Vladimir Tomin](https://www.behance.net/myaka)


![divisoria](https://user-images.githubusercontent.com/24854541/94681772-5a009800-02fa-11eb-8ff8-29f9fad9b18f.png)

**[GNU AGPL v3.0](https://www.gnu.org/licenses/agpl-3.0.html)**
