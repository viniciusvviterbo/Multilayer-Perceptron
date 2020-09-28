#include <iostream>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

/*
 * Place the given x number between 0 and 1
 * according to the distance it has from the minimum,
 * in comparison to the distance between min and max
 */
double normalizedValue(double x, double minValue, double maxValue)
{
    return (x - minValue) / (maxValue - minValue);
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

int main()
{

    // Declares and reads the main values of the dataset input
    int datasetLength, inputLength, outputLength;

    cin >> datasetLength >> inputLength >> outputLength;

    int nCol = inputLength + outputLength;
    // Alocates memory for the dataset reading
    double **dataset = (double **)calloc(datasetLength, sizeof(double *));

    // Reads the dataset data
    for (int i = 0; i < datasetLength; i++)
    {
        dataset[i] = (double *)calloc(nCol, sizeof(double));

        for (int j = 0; j < nCol; j++)
        {
            cin >> dataset[i][j];
        }
    }

    //Function that normalizes the dataset, placing every value between 0 and 1
    //for a better comparison, making it better for the multilayer perceptron neural
    //network converge the error to the given threshold
    normalizeDataset(dataset, datasetLength, nCol);

    cout << datasetLength << " " << inputLength << " " << outputLength << endl << endl;

    //printing back all the normalized dataset (the idea is to redirect the standard output to a file)
    for (int i = 0; i < datasetLength; i++)
    {
        for (int j = 0; j < nCol; j++)
        {
            printf("%g ", dataset[i][j]);
        }
        cout << endl;
    }

    return 0;
}