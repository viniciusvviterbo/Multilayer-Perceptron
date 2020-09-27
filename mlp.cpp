#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <list>
#include <time.h>

#define PATTERN_FILE "pattern.in" //File name of the pattern to be learnt by the neural network

using namespace std;

//Miscelaneous usefull functions and procedures
#pragma region Util
// Converts a list of tuples to a matrix used as a pattern for the entries
double **listToPatternMatrix(list<double *> listToConvert, int rowCount, int columnCount)
{
    list<double *>::iterator it;
    int i, j;
    double **pattern = (double **)calloc(rowCount, sizeof(double *));
    for (i = 0, it = listToConvert.begin(); i < rowCount; i++, it++)
    {
        pattern[i] = (double *)calloc(columnCount + 1, sizeof(double));
        double *row = *it;
        for (j = 0; j < columnCount; j++)
        {
            pattern[i][j] = row[j];
        }
        pattern[i][columnCount] = 1;
    }
    return pattern;
}

// Reads from the pattern file and returns a parsed list with the entries
double **readPatternFile(int inputLength)
{
    list<double *> pattern;                       // Initiates the entries/input pattern
    double *buffer;                               // Initiates the buffer to inform the input matriz values
    char *tmpLine;                                // Initiates the buffer for reading the file lines
    FILE *johnLennon = fopen(PATTERN_FILE, "rb"); // Instantiates the file pointer

    if (johnLennon == NULL) // Tests readability of file
        cerr << "Error opening file";
    else // If file is readable, excutes
    {
        char *line = NULL;                               // Declares the line for each iteration
        size_t len = 0;                                  // Declares the size of the line for each iteration
        while ((getline(&line, &len, johnLennon)) != -1) // While the file has lines to be read, executes
        {
            buffer = (double *)calloc(inputLength, sizeof(double)); // Clears the buffer for each iteration

            for (int i = 0; i < inputLength; i++) // Reads the relevant values (for now) of the line
            {
                tmpLine = strtok(line, " "); // Splits the line by spaces
                buffer[i] = atof(tmpLine);   // Converts the read value to float/double
            }

            pattern.push_back(buffer); // Appends the values to the buffer
        }
    }
    fclose(johnLennon);
    return listToPatternMatrix(pattern, pattern.size(), inputLength);
}

// Multiply a matrix by a vector and return the result
double *matrixVectorMultiplication(double **matrix, double *vec, int rowCountMatrix, int columnCountMatrix, int rowCountVector)
{
    double *result;
    result = (double *)calloc(rowCountMatrix, sizeof(double));

    if (columnCountMatrix == rowCountVector + 1)
    {
        for (int i = 0; i < rowCountMatrix; i++) 
        {
            for (int j = 0; j < rowCountVector; j++) 
            {
                result[i] += matrix[i][j] * vec[j];
            }
            result[i] -= matrix[i][rowCountVector];
        }
    }
    return result;
}

// Multiply a vector by a matrix and return the result
double *vectorMatrixMultiplication(double *vec, double **matrix, int columnCountVector, int rowCountMatrix, int columnCountMatrix)
{
    double *result;
    result = (double *)calloc(columnCountMatrix, sizeof(double));
    if (columnCountVector == rowCountMatrix)
    {
        for (int i = 0; i < columnCountMatrix; i++)
        {
            for (int j = 0; j < columnCountVector; j++)
            {
                result[i] += vec[j] * matrix[j][i];
            }
        }
    }

    return result;
}

// Multiply a vector by a vector and return the result which is a (lenghtA x lengthB) matrix
double **vectorMultiplication(double *a, double *b, int lengthA, int lengthB)
{
    double **result;
    result = (double **)calloc(lengthA, sizeof(double *));

    for (int i = 0; i < lengthA; i++)
    {
        result[i] = (double *)calloc(lengthB + 1, sizeof(double));
        for (int j = 0; j < lengthB; j++)
        {
            result[i][j] += a[i] * b[j];
        }
        result[i][lengthB] = -a[i];
    }
    return result;
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

    double **hiddenLayer; // Array storing weights and thresholds of each neuron in the hidden layer
    double **outputLayer; // Array storing weights and thresholds of each neuron in the output layer

#pragma endregion

public:
#pragma region Constructor
    MLP()
    {
        this->inputLength = 2;
        this->hiddenLayerLength = 2;
        this->outputLayerLength = 1;

        this->hiddenLayer = (double **)calloc(this->hiddenLayerLength, sizeof(double *));
        for (int i = 0; i < this->hiddenLayerLength; i++)
        {
            this->hiddenLayer[i] = (double *)calloc(this->inputLength + 1, sizeof(double));
        }

        this->outputLayer = (double **)calloc(this->outputLayerLength, sizeof(double *));
        for (int i = 0; i < this->outputLayerLength; i++)
        {
            this->outputLayer[i] = (double *)calloc(this->hiddenLayerLength + 1, sizeof(double));
        }

        this->populateNeurons();
    }

    MLP(int inputLength, int hiddenLayerLength, int outputLayerLength)
    {
        this->inputLength = inputLength;
        this->hiddenLayerLength = hiddenLayerLength;
        this->outputLayerLength = outputLayerLength;

        this->hiddenLayer = (double **)calloc(this->hiddenLayerLength, sizeof(double *));
        for (int i = 0; i < this->hiddenLayerLength; i++)
        {
            this->hiddenLayer[i] = (double *)calloc(this->inputLength + 1, sizeof(double));
        }

        this->outputLayer = (double **)calloc(this->outputLayerLength, sizeof(double *));
        for (int i = 0; i < this->outputLayerLength; i++)
        {
            this->outputLayer[i] = (double *)calloc(this->hiddenLayerLength + 1, sizeof(double));
        }

        this->populateNeurons();
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
                this->hiddenLayer[i][j] = ((double)rand() / (RAND_MAX)) - 0.5;
            }
        }

        for (int i = 0; i < this->outputLayerLength; i++)
        {
            for (int j = 0; j < this->hiddenLayerLength + 1; j++)
            {
                // Generates a random number between -0.5 and 0.5 for each weight and threshold of a neuron
                this->outputLayer[i][j] = ((double)rand() / (RAND_MAX)) - 0.5;
            }
        }
    }

    // Given the weighted values received by the neuron, this function is executed in order for the neuron to find a result
    void activation(double *net, int netLength)
    {
        for (int i = 0; i < netLength; i++)
        {
            net[i] = 1.0 / (1 + exp(-net[i]));
        }
    }

    // Calculates the derivatives of the activated values
    double *dNet(double *f_net, int f_netLength)
    {
        double *df_dnet = (double *)calloc(f_netLength, sizeof(double));
        for (int i = 0; i < f_netLength; i++)
        {
            df_dnet[i] = (f_net[i] * (1 - f_net[i]));
        }
        return df_dnet;
    }

    // Receives one input from the dataset and obtains a response from the network
    State forward(double *input)
    {
        // Finds the net value (sum of weighted inputs) received by the neuron in the hidden layer
        double *netHiddenLayer = matrixVectorMultiplication(this->hiddenLayer, input, this->hiddenLayerLength, this->inputLength + 1, this->inputLength);
        
        // Finds the values calculated by the hidden layer
        this->activation(netHiddenLayer, this->hiddenLayerLength);

        // Output layer

        // Finds the net value (sum of weighted inputs) received by the neuron in the output layer
        double *netOutputLayer = matrixVectorMultiplication(this->outputLayer, netHiddenLayer, this->outputLayerLength, this->hiddenLayerLength + 1, this->hiddenLayerLength);
        
        // Finds the values calculated by the hidden layer
        this->activation(netOutputLayer, this->outputLayerLength);

        // Declares and instantiates the current state of the network
        State state;
        state.fNetHidden = netHiddenLayer;
        state.fNetOutput = netOutputLayer;

        return state;
    }

    // Trains the neural network based on the mistakes made
    void backPropagation(double **input, double **output, int datasetLength, double trainingRate, double threshold)
    {

        int count = 0;
        //vectors used in the method
        double *Xp, *Yp, *errors, *dNetOutput, *deltaOutput, *dNetHidden, *deltaHidden, *outputDerivative;
        //matrices used in the method
        double **outputLayerCorrection, **hiddenLayerCorrection;
        State results;

        double squaredError = 2 * threshold;
        // Executes the loop while the error acceptance is not satiated
        while (squaredError > threshold)
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
                // Declares a vector for the calculated errors
                errors = (double *)calloc(this->outputLayerLength, sizeof(double));
                //cout << "Errors: [";
                // Calculates the error for each value obtained
                for (int i = 0; i < this->outputLayerLength; i++)
                {
                    errors[i] = Yp[i] - results.fNetOutput[i];
                    squaredError += pow(errors[i], 2);
                    //cout << errors[i] << " ";
                }
                //cout << "]\n";

                // Finds the derivative of the line that represents the error
                dNetOutput = dNet(results.fNetOutput, this->outputLayerLength);

                // Declares a vector for the calculated derivatives
                deltaOutput = (double *)calloc(this->outputLayerLength, sizeof(double));
                // Calculates the derivative for each error stored
                for (int i = 0; i < this->outputLayerLength; i++)
                {
                    deltaOutput[i] = errors[i] * dNetOutput[i];
                }
#pragma endregion

#pragma region Hidden layer manipulation
                outputDerivative = vectorMatrixMultiplication(deltaOutput, this->outputLayer, this->outputLayerLength, this->outputLayerLength , this->hiddenLayerLength);
                
                // Declares a vector for the calculated derivatives
                deltaHidden = (double *)calloc(this->hiddenLayerLength, sizeof(double));

                // Finds the derivative of the line that represents the error
                dNetHidden = dNet(results.fNetHidden, this->hiddenLayerLength);
               
                // Calculates the derivative for each error stored
                for (int i = 0; i < this->hiddenLayerLength; i++)
                {
                    deltaHidden[i] = outputDerivative[i] * dNetHidden[i];
                }
#pragma endregion

#pragma region Effective training
                outputLayerCorrection = vectorMultiplication(deltaOutput, results.fNetHidden, this->outputLayerLength, this->hiddenLayerLength);
    
                for (int i = 0; i < this->outputLayerLength; i++)
                {
                    for (int j = 0; j < this->hiddenLayerLength + 1; j++)
                    {
                        this->outputLayer[i][j] += trainingRate * outputLayerCorrection[i][j];
                    }
                }

                hiddenLayerCorrection = vectorMultiplication(deltaHidden, Xp, this->hiddenLayerLength, this->inputLength);
                for (int i = 0; i < this->hiddenLayerLength; i++)
                {
                    for (int j = 0; j < this->inputLength + 1; j++)
                    {
                        this->hiddenLayer[i][j] += trainingRate * hiddenLayerCorrection[i][j];
                    }
                }
#pragma endregion
            }

            squaredError /= datasetLength;
            count++;
            //free all used memory
            free(errors);
            free(dNetOutput);
            free(deltaOutput);
            free(dNetHidden);
            free(deltaHidden);
            free(outputDerivative);
        }
        
        cout << "Number of epochs = " << count << endl;
    }

#pragma endregion
};

int main()
{
    // Declares the MLP class
    MLP *mlp = new MLP(2, 2, 1);

    double **input = (double **)calloc(4, sizeof(double *));
    double **output = (double **)calloc(4, sizeof(double *));
    for (int i = 0; i < 4; i++)
    {
        input[i] = (double *)calloc(3, sizeof(double));
        output[i] = (double *)calloc(1, sizeof(double));
    }

    input[0][0] = 0;
    input[0][1] = 0;
    input[1][0] = 0;
    input[1][1] = 1;
    input[2][0] = 1;
    input[2][1] = 0;
    input[3][0] = 1;
    input[3][1] = 1;
    output[0][0] = 0;
    output[1][0] = 1;
    output[2][0] = 1;
    output[3][0] = 0;

    mlp->backPropagation(input, output, 4, 0.2, 1e-5);

    State state;

    double *Xp = (double *)calloc(2, sizeof(double));
    Xp[0] = 0;
    Xp[1] = 0;
    state = mlp->forward(Xp);

    cout << "0 XOR 0 = " << round(state.fNetOutput[0]) << '\n';

    Xp[0] = 0;
    Xp[1] = 1;
    state = mlp->forward(Xp);

    cout << "0 XOR 1 = " << round(state.fNetOutput[0]) << '\n';

    Xp[0] = 1;
    Xp[1] = 0;
    state = mlp->forward(Xp);

    cout << "1 XOR 0 = " << round(state.fNetOutput[0]) << '\n';

    Xp[0] = 1;
    Xp[1] = 1;
    state = mlp->forward(Xp);

    cout << "1 XOR 1 = " << round(state.fNetOutput[0]) << '\n';

    return (0);
}