#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <list>

using namespace std;

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
    }
#pragma endregion

#pragma region Forward
    //
    double activation(double net)
    {
        return 1 / (1 + exp(-net));
    }

    //
    double dNet(double f_net)
    {
        return (f_net * (1 - f_net));
    }

    // Reads the dataset and executes the forward motion in the neural network
    void forward()
    {
        list<double*> pattern; // Initiates the entries/input pattern
        double *buffer; // Initiates the buffer to inform the input matriz values
        char *tmpLine; // Initiates the buffer for reading the file lines
        FILE *johnLennon = fopen("pattern_file.in", "rb"); // Instantiates the file pointer

        if (johnLennon == NULL) // Tests readability of file
            cerr << "Error opening file";
        else // If file is readable, excutes
        {
            char *line = NULL; // Declares the line for each iteration
            size_t len = 0; // Declares the size of the line for each iteration
            while ((getline(&line, &len, johnLennon)) != -1) // While the file has lines to be read, executes
            {
                buffer = (double *)calloc(this->inputLength, sizeof(double)); // Clears the buffer for each iteration
                
                for(int i = 0; i < this->inputLength; i++) // Reads the relevant values (for now) of the line
                {
                    tmpLine = strtok(line, " "); // Splits the line by spaces
                    buffer[i] = atof(tmpLine); // Converts the read value to float/double
                }

                pattern.push_back(buffer); // Appends the values to the buffer
            }

            // PATTERN IS COMPLETE BY NOW
        }
        fclose(johnLennon);
    }
#pragma endregion
};

int main()
{
    return (0);
}