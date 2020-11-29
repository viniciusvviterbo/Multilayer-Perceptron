/*
 * Grupo: Lincoln Antunes, Mateus Alberto e Vinícius Viterbo
 * Para cada versão, rodamos 5 vezes e tiramos uma média do tempo
 * Tempo sequencial: 35,0124 segundos
 * Tempo paralelo openMP 2 threads: 24,6672 segundos
 *      speedup = 1,42
 * Tempo paralelo openMP 4 threads: 22,6344 segundos
 *      speedup = 1,55
 * Tempo paralelo openMP 8 threads: 23,1651 segundos
 *      speedup = 1,51
 * Obs.: o tempo paralelo com 8 threads pode ter ficado um pouco
 *       pior do que com 4 threads por não ter 8 threads disponíveis
 *       para executar a todo momento 
 *
 * Tempo CUDA: 3+ minutos
 */

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <list>
#include <time.h>
#include <stdio.h>

using namespace std;

//Miscelaneous usefull functions and procedures
#pragma region Util

//CUDA KERNELS

// Multiply a matrix by a vector and return the result
__global__ void cuda_matrixVectorMultiplication(double *result, double *matrix, double *vec, int rowCountMatrix, int columnCountMatrix, int rowCountVector)
{
    //printf("kernel cuda\n");
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;

    if (row < rowCountMatrix && col < columnCountMatrix)
    {
        for (int i = 0; i < rowCountVector; i++)
        {
            sum += matrix[row * columnCountMatrix + i] * vec[i];
        }
        result[row] = sum - matrix[row * columnCountMatrix + rowCountVector];
    }
}

// Multiply a vector by a matrix and return the result
__global__ void cuda_vectorMatrixMultiplication(double *result, double *vec, double *matrix, int columnCountVector, int rowCountMatrix, int columnCountMatrix)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0;

    if (row < rowCountMatrix && col < columnCountMatrix)
    {
        for (int i = 0; i < columnCountVector; i++)
        {
            sum += vec[i] * matrix[i * columnCountMatrix + col];
        }
        result[col] = sum;
    }
}

// Given the weighted values received by the neuron, this function is executed in order for the neuron to find a result
__global__ void cuda_activation(double *net, int netLength)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < netLength) {
        net[i] = 1.0 / (1 + exp(-net[i]));
    }
}

// Multiply a matrix by a vector and return the result
void matrixVectorMultiplication(double *result, double *matrix, double *vec, int rowCountMatrix, int columnCountMatrix, int rowCountVector)
{
    double sum;
    if (columnCountMatrix == rowCountVector + 1)
    {
        for (int i = 0; i < rowCountMatrix; i++)
        {
            sum = 0;
            for (int j = 0; j < rowCountVector; j++)
            {
                sum += matrix[i * columnCountMatrix + j] * vec[j];
            }
            result[i] = sum - matrix[i * columnCountMatrix + rowCountVector];
        }
    }
}

// Multiply a vector by a matrix and return the result
void vectorMatrixMultiplication(double *result, double *vec, double *matrix, int columnCountVector, int rowCountMatrix, int columnCountMatrix)
{
    double sum;
    if (columnCountVector == rowCountMatrix)
    {
        for (int i = 0; i < columnCountMatrix; i++)
        {
            sum = 0;
            for (int j = 0; j < columnCountVector; j++)
            {
                sum += vec[j] * matrix[j * columnCountMatrix + i];
            }
            result[i] = sum;
        }
    }
}

//Function that compares two given vectors and returns true if all of their elements are the same
bool equalVectors(double *a, double *b, int length)
{
        for(int i = 0; i < length; i++)
        {
                if(a[i] != round(b[i])) return false;
        }
        return true;
}

//Put 0 to every element of given array
void clearArray(double *array, int size) {
    for(int i = 0; i < size; i++) {
        array[i] = 0;
    }
}


#pragma endregion

// Representation of the values with a given input at each step in the forward motion
struct State
{
    double *fNetHidden; // Stores the values given by the neurons on the hidden layer
    double *fNetOutput; // Stores the values given by the neurons on the output layer
} typedef State;

// Multi-Layer Perceptron neural network class
class MLP
{
#pragma region Properties
private:
    int inputLength;       // Number of inputs for each neuron
    int hiddenLayerLength; // Number of neurons in the hidden layer
    int outputLayerLength; // Number of neurons in the output layer

    double *hiddenLayer; // Array storing weights and thresholds of each neuron in the hidden layer
    double *outputLayer; // Array storing weights and thresholds of each neuron in the output layer

    //Net
    double *netHiddenLayer;
    double *netOutputLayer;

    //--------- CUDA variables -------------
    //Sizes
    int size_input, size_hiddenLayer, size_netHiddenLayer, size_outputLayer, size_netOutputLayer, size_deltaOutput, size_outputDerivative;

    //Device(GPU) Arrays
    double *d_hiddenLayer, *d_outputLayer;
    double *d_input, *d_netHiddenLayer, *d_netOutputLayer, *d_outputDerivative, *d_deltaOutput;

    //block and grid dimensions
    dim3 dimGridH, dimGridActivationH, dimGridActivationO, dimGridO;
    dim3 dimBlock1D, dimBlock2D;

#pragma endregion

public:
#pragma region Constructors and Destructors
    MLP()
    {
        this->inputLength = 2;
        this->hiddenLayerLength = 2;
        this->outputLayerLength = 1;

        this->hiddenLayer = (double *)calloc(this->hiddenLayerLength * this->inputLength + 1, sizeof(double));

        this->netHiddenLayer = (double*) calloc(this->hiddenLayerLength, sizeof(double));

        this->outputLayer = (double *)calloc(this->outputLayerLength * this->hiddenLayerLength + 1, sizeof(double));

        this->netOutputLayer = (double*) calloc(this->outputLayerLength, sizeof(double));

        this->populateNeurons();

        this->initializeCUDA();
    }

    MLP(int inputLength, int hiddenLayerLength, int outputLayerLength)
    {
        this->inputLength = inputLength;
        this->hiddenLayerLength = hiddenLayerLength;
        this->outputLayerLength = outputLayerLength;

        this->hiddenLayer = (double *)calloc(this->hiddenLayerLength * this->inputLength + 1, sizeof(double));

        this->netHiddenLayer = (double*) calloc(this->hiddenLayerLength, sizeof(double));

        this->outputLayer = (double *)calloc(this->outputLayerLength * this->hiddenLayerLength + 1, sizeof(double));

        this->netOutputLayer = (double*) calloc(this->outputLayerLength, sizeof(double));

        this->populateNeurons();

        this->initializeCUDA();
    }

    //Destructor
    ~MLP() {
        delete hiddenLayer;
        delete outputLayer;
        delete netHiddenLayer;
        delete netOutputLayer;

        cudaFree(d_input);
        cudaFree(d_hiddenLayer);
        cudaFree(d_outputLayer);
        cudaFree(d_netHiddenLayer);  
        cudaFree(d_netOutputLayer);  
        cudaFree(d_outputDerivative);
        cudaFree(d_deltaOutput);
    }
#pragma endregion

#pragma region Methods
    // Initialize the values for weights and thresholds for each neuron
    void populateNeurons()
    {
        srand(time(NULL));

        for (int i = 0; i < this->hiddenLayerLength; i++)
        {
            for (int j = 0; j < this->inputLength + 1; j++)
            {
                // Generates a random number between -0.5 and 0.5 for each weight and threshold of a neuron
                this->hiddenLayer[i*(this->inputLength + 1) + j] = ((double)rand() / (RAND_MAX)) - 0.5;
            }
        }

        for (int i = 0; i < this->outputLayerLength; i++)
        {
            for (int j = 0; j < this->hiddenLayerLength + 1; j++)
            {
                // Generates a random number between -0.5 and 0.5 for each weight and threshold of a neuron
                this->outputLayer[i * (this->hiddenLayerLength + 1) + j] = ((double)rand() / (RAND_MAX)) - 0.5;
            }
        }
    }

    //Method for initializing CUDA variables
    void initializeCUDA() {
        //Initializing Sizes
        size_input = this->inputLength * sizeof(double);
        size_hiddenLayer = (this->inputLength + 1) * this->hiddenLayerLength * sizeof(double);
        size_netHiddenLayer = this->hiddenLayerLength * sizeof(double);
        size_outputLayer = (this->hiddenLayerLength + 1) * this->outputLayerLength * sizeof(double);
        size_netOutputLayer = this->outputLayerLength * sizeof(double);
        size_deltaOutput = this->outputLayerLength * sizeof(double);
        size_outputDerivative = this->hiddenLayerLength * sizeof(double);
        
        //Initializing block and grid sizes
        int block_size1D = 64;
        int block_size2D = 8;

        int grid_sizeActivationH = (this->hiddenLayerLength - 1) / block_size1D + 1;
        int grid_sizeActivationO = (this->outputLayerLength - 1) / block_size1D + 1;

        int row_grid_sizeH = (this->hiddenLayerLength - 1) / block_size2D + 1;
        int col_grid_sizeH = this->inputLength / block_size2D + 1;
        int row_grid_sizeO = (this->outputLayerLength - 1) / block_size2D + 1;
        int col_grid_sizeO = (this->hiddenLayerLength) / block_size2D + 1;

        //Initializing CUDA Dimensions
        dimBlock1D.x = block_size1D;
        dimBlock2D.x = block_size2D;
        dimBlock2D.y = block_size2D;

        dimGridActivationH.x = grid_sizeActivationH;
        dimGridActivationO.x = grid_sizeActivationO;
        dimGridH.x = row_grid_sizeH;
        dimGridH.y = col_grid_sizeH;
        dimGridO.x = row_grid_sizeO;
        dimGridO.y = col_grid_sizeO;

        //GPU Memory Allocations
        cudaMalloc((void**) &d_input, size_input);
        cudaMalloc((void**) &d_hiddenLayer, size_hiddenLayer);
        cudaMalloc((void**) &d_netHiddenLayer, size_netHiddenLayer);
        cudaMalloc((void**) &d_outputLayer, size_outputLayer);
        cudaMalloc((void**) &d_netOutputLayer, size_netOutputLayer);
        cudaMalloc((void**) &d_deltaOutput, size_deltaOutput);
        cudaMalloc((void**) &d_outputDerivative, size_outputDerivative);

        //Copying memory to device
        cudaMemcpy(d_hiddenLayer, this->hiddenLayer, size_hiddenLayer, cudaMemcpyHostToDevice);
        cudaMemcpy(d_outputLayer, this->outputLayer, size_outputLayer, cudaMemcpyHostToDevice);
    }

    // Given the weighted values received by the neuron, this function is executed in order for the neuron to find a result
    void activation(double *net, int netLength)
    {
        for (int i = 0; i < netLength; i++)
        {
            net[i] = 1.0 / (1 + exp(-net[i]));
        }
    }

    // Calculates the derivatives of the activated values and places the result in df_dnet
    void dNet(double *df_dnet, double *f_net, int f_netLength)
    {
        for (int i = 0; i < f_netLength; i++)
        {
            df_dnet[i] = (f_net[i] * (1 - f_net[i]));
        }
    }

    // Receives one input from the dataset and obtains a response from the network
    State forward(double *input)
    {   
        //zeroing nets
        clearArray(netHiddenLayer, hiddenLayerLength);
        clearArray(netOutputLayer, outputLayerLength);

        cudaMemcpy(d_input, input, size_input, cudaMemcpyHostToDevice);

        // Finds the net value (sum of weighted inputs) received by the neuron in the hidden layer
        cuda_matrixVectorMultiplication<<<dimGridH, dimBlock2D>>>(d_netHiddenLayer, d_hiddenLayer, d_input, this->hiddenLayerLength, this->inputLength + 1, this->inputLength);      
        
        //Activation of netHiddenLayer, producing df_dnetH
        cuda_activation<<<dimGridActivationH, dimBlock1D>>>(d_netHiddenLayer, this->hiddenLayerLength);
        cudaMemcpy(netHiddenLayer, d_netHiddenLayer, size_netHiddenLayer, cudaMemcpyDeviceToHost);
        
        // Output layer
        // Finds the net value (sum of weighted inputs) received by the neuron in the output layer
        cuda_matrixVectorMultiplication<<<dimGridO, dimBlock2D>>>(d_netOutputLayer, d_outputLayer, d_netHiddenLayer, this->outputLayerLength, this->hiddenLayerLength + 1, this->hiddenLayerLength);

        //Activation of netOutputLayer, producing df_dnetO
        cuda_activation<<<dimGridActivationO, dimBlock1D>>>(d_netOutputLayer, this->outputLayerLength);

        // Declares and instantiates the current state of the network
        cudaMemcpy(netOutputLayer, d_netOutputLayer, size_netOutputLayer, cudaMemcpyDeviceToHost);
        State state;
        state.fNetHidden = netHiddenLayer;
        state.fNetOutput = netOutputLayer;

        return state;
   }

    // Trains the neural network based on the mistakes made
    void backPropagation(double **input, double **output, int datasetLength, double trainingRate, double threshold)
    {
        int count = 0;
        // Vectors used in the method
        double *Xp, *Yp, *errors, *dNetOutput, *deltaOutput, *dNetHidden, *deltaHidden, *outputDerivative;
        // Matrices used in the method
        State results;

        //Allocates memory for the arrays
        // Declares a vector for the calculated errors
        errors = (double *)calloc(this->outputLayerLength, sizeof(double));
        // Declares the vector for the line derivative that represents the error for the output layer
        dNetOutput = (double *)calloc(this->outputLayerLength, sizeof(double));
        // Declares a vector for the calculated derivatives
        deltaOutput = (double *)calloc(this->outputLayerLength, sizeof(double));
        outputDerivative = (double *)calloc(this->hiddenLayerLength, sizeof(double));

        // Declares the vector for the line derivative that represents the error for the hidden layer
        dNetHidden = (double *)calloc(this->hiddenLayerLength, sizeof(double));
        // Declares a vector for the calculated derivatives
        deltaHidden = (double *)calloc(this->hiddenLayerLength, sizeof(double));

        double squaredError = 2 * threshold;
        // Executes the loop while the error acceptance is not satiated
        while (squaredError > threshold && count < 1)
        {
            squaredError = 0;

            for (int p = 0; p < datasetLength; p++)
            {
                // Extracts input pattern
                Xp = input[p];
                // Extracts output pattern
                Yp = output[p];

                // Obtains the results given by the network
                results = this->forward(Xp);

                #pragma region Output layer manipulation
                
                // Calculates the error for each value obtained
                for (int i = 0; i < this->outputLayerLength; i++)
                {
                    errors[i] = Yp[i] - results.fNetOutput[i];
                    squaredError += pow(errors[i], 2);
                }

                // Finds the derivative of the line that represents the error
                dNet(dNetOutput, results.fNetOutput, this->outputLayerLength);
                
                // Calculates the derivative for each error stored
                for (int i = 0; i < this->outputLayerLength; i++)
                {
                    deltaOutput[i] = errors[i] * dNetOutput[i];
                }
#pragma endregion

#pragma region Hidden layer manipulation

                // Alocates memory for vector and matrix multiplication on the device
                cudaMemcpy(d_deltaOutput, deltaOutput, size_deltaOutput, cudaMemcpyHostToDevice);

                // Finds the net value (sum of weighted inputs) received by the neuron in the output layer
                cuda_vectorMatrixMultiplication<<<dimGridO, dimBlock2D>>>(d_outputDerivative, d_deltaOutput, d_outputLayer, this->outputLayerLength, this->outputLayerLength, this->hiddenLayerLength);
                cudaMemcpy(outputDerivative, d_outputDerivative, size_outputDerivative, cudaMemcpyDeviceToHost);

                //vectorMatrixMultiplication(outputDerivative, deltaOutput, this->outputLayer, this->outputLayerLength, this->outputLayerLength, this->hiddenLayerLength);

                // Finds the derivative of the line that represents the error
                dNet(dNetHidden, results.fNetHidden, this->hiddenLayerLength);

                // Calculates the derivative for each error stored
                for (int i = 0; i < this->hiddenLayerLength; i++)
                {
                    deltaHidden[i] = outputDerivative[i] * dNetHidden[i];
                }

                #pragma endregion

                #pragma region Effective training

                //Training the output layer
                for (int i = 0; i < this->outputLayerLength; i++)
                {
                    for (int j = 0; j < this->hiddenLayerLength; j++)
                    {
                        this->outputLayer[i * (this->hiddenLayerLength + 1) + j] += trainingRate * deltaOutput[i] * results.fNetHidden[j];
                    }
                    this->outputLayer[i * (this->hiddenLayerLength + 1) + this->hiddenLayerLength] -= trainingRate * deltaOutput[i];
                }
                cudaMemcpy(d_outputLayer, this->outputLayer, size_outputLayer, cudaMemcpyHostToDevice);

                //Training the hidden layer
                for (int i = 0; i < this->hiddenLayerLength; i++)
                {
                    for (int j = 0; j < this->inputLength; j++)
                    {
                        this->hiddenLayer[i * (this->inputLength + 1) + j] += trainingRate * deltaHidden[i] * Xp[j];
                    }
                    this->hiddenLayer[i * (this->inputLength + 1) + inputLength] -= trainingRate * deltaHidden[i];
                }
                cudaMemcpy(d_hiddenLayer, this->hiddenLayer, size_hiddenLayer, cudaMemcpyHostToDevice);
                #pragma endregion
            }

            squaredError /= datasetLength;
            count++;
        }
        // Clear all used memory
        delete errors;
        delete deltaOutput;
        delete deltaHidden;
        delete dNetOutput;
        delete dNetHidden;
        delete outputDerivative;
    }

    #pragma endregion
};

int main(int argc, char *argv[])
{
    // Declares and reads the main values of the dataset input
    int datasetLength, inputLength, outputLength, hiddenLength = atoi(argv[1]);
    double trainingRate = atof(argv[2]), threshold = atof(argv[3]);

    cin >> datasetLength >> inputLength >> outputLength;

    // Alocates memory for the dataset reading
    double **input = (double **)calloc(datasetLength, sizeof(double *));
    double **output = (double **)calloc(datasetLength, sizeof(double *));

    // Reads the dataset data
    for (int i = 0; i < datasetLength; i++)
    {
        input[i] = (double *)calloc(inputLength, sizeof(double));
        output[i] = (double *)calloc(outputLength, sizeof(double));

        for (int j = 0; j < inputLength; j++)
        {
            cin >> input[i][j];
        }

        for (int j = 0; j < outputLength; j++)
        {
            cin >> output[i][j];
        }
    }

    // Declares the MLP network class
    MLP *mlp = new MLP(inputLength, hiddenLength, outputLength);

    // Executes the neural network training
    mlp->backPropagation(input, output, datasetLength/2, trainingRate, threshold);


    #pragma region Testing
    State state;

    double *Xp;

    int errorCount = 0;
    
    for(int i = datasetLength/2; i < datasetLength; i++)
    {
        Xp = input[i];
        state = mlp->forward(Xp);
        if(!equalVectors(output[i], state.fNetOutput, outputLength)) errorCount++;
    }

    int nTests = datasetLength/2;
    cout << "Number of tests = " << nTests << endl;
    int nSucc = nTests - errorCount;
    cout << "Number of succesful answers = " << nSucc << endl;
    cout << "Number of errors = " << errorCount << endl;
    cout << "Accuracy = " << (double)nSucc/nTests * 100 << "%\n"; 
    #pragma endregion

    //delete mlp;

    return (0);
}
