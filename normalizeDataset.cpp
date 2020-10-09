#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace std;

/*
 * Place the given x number between 0 and 1
 * according to the distance it has from the minimum,
 * in comparison to the distance between min and max
 */
double normalizedValue(double x, double minValue, double maxValue)
{
    if ((maxValue - minValue) == 0) return 0;
    else return (x - minValue) / (maxValue - minValue);
}

/*
 * Find the minimum and maximum values of each column,
 * updating the given arrays by reference
 */
void minMaxValues(double **dataset, double *minValues, double *maxValues, int nRow, int nCol)
{
    //Initiating every position of the min and max arrays
    //with a number from the dataset, guaranteeing the
    //min and max values at the end of the algorithm
    for(int j = 0; j < nCol; j++) 
    {
        minValues[j] = dataset[0][j];
        maxValues[j] = dataset[0][j];
    }

    for(int i = 0; i < nRow; i++)
    {
        for(int j = 0; j < nCol; j++)
        {
            //checks if the current number is less than
            //the minValue of the column, updating it if true
            if(dataset[i][j] < minValues[j]) 
            {
                minValues[j] = dataset[i][j];
            } 
            else 
            {
                //checks if the current number is greater than
                //the maxValue of the column, updating it if true
                if(dataset[i][j] > maxValues[j])
                {
                    maxValues[j] = dataset[i][j];
                }
            }
        }
    }
}

/*
 * Normalizes the dataset, placing every value between 0 and 1, accordig
 * to the min and max values of each column
*/
void normalizeDataset(double **dataset, int nRow, int nCol) 
{
    double *minValues = (double*) calloc(nCol, sizeof(double));
    double *maxValues = (double*) calloc(nCol, sizeof(double));
    minMaxValues(dataset, minValues, maxValues, nRow, nCol);
    for(int i = 0; i < nRow; i++)
    {
        for(int j = 0; j < nCol; j++)
        {
            //gets the number between 0 and 1
            dataset[i][j] = normalizedValue(dataset[i][j], minValues[j], maxValues[j]);
        }
    }
}

//Generates a completele ramdomized and normalized dataset
void randomDataset(int datasetLength, int inputLength, int outputLength)
{
    cout << datasetLength << " " << inputLength << " " << outputLength << endl << endl;
    srand(time(NULL));
    double val;
    for(int i = 0; i < datasetLength; i++) 
    {
        for(int j = 0; j < inputLength; j++)
        {
            val = ((double)rand() / (RAND_MAX));
            cout << val << " ";
        }
        for(int j = 0; j < outputLength; j++)
        {
            val = round(((double)rand() / (RAND_MAX)));
            cout << val << " ";
        }
        cout << endl;
    }
}


int main()
{

    // Declares and reads the main values of the dataset input
    int datasetLength, inputLength, outputLength;

    cin >> datasetLength >> inputLength >> outputLength;

    // Alocates memory for the dataset reading
    double **dataset = (double **)calloc(datasetLength, sizeof(double *));

    double ncol = inputLength + outputLength;
    // Reads the dataset data
    for (int i = 0; i < datasetLength; i++)
    {
        dataset[i] = (double *)calloc(ncol, sizeof(double));

        for (int j = 0; j < ncol; j++)
        {
            cin >> dataset[i][j];
        }
    }
    
    /*datasetLength = 700;
    inputLength = 128;
    outputLength = 1;

    randomDataset(700, 60, 6);
    //randomDataset(4, 2, 1);
    */

    //Function that normalizes the dataset, placing every value between 0 and 1
    //for a better comparison, making it better for the multilayer perceptron neural
    //network converge the error to the given threshold
    normalizeDataset(dataset, datasetLength, ncol);

    cout << datasetLength << " " << inputLength << " " << outputLength << endl << endl;
    
    //printing back all the normalized dataset (the idea is to redirect the standard output to a file)
    for (int i = 0; i < datasetLength; i++)
    {
        for (int j = 0; j < ncol; j++)
        {
            printf("%g ", dataset[i][j]);
        }
        cout << endl;
    }

    return 0;
}