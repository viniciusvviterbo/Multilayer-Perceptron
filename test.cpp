#include <stdlib.h>
#include <iostream>

using namespace std;

double *matrixVectorMultiplication(double **matrix, double *vec, int rowCountMatrix, int columnCountMatrix, int rowCountVector)
{
    double *result =  (double*) calloc(rowCountMatrix, sizeof(double));
    double sum;
    if (columnCountMatrix == rowCountVector + 1)
    {
        //#pragma omp parallel for num_threads(2)
        for (int i = 0; i < rowCountMatrix; i++)
        {
            sum = 0;
            #pragma omp parallel for reduction(+: sum) num_threads(2)
            for (int j = 0; j < rowCountVector; j++)
            {
                sum += matrix[i][j] * vec[j];
            }
            result[i] = sum - matrix[i][rowCountVector];
        }
    }
    return result;
}

// Multiply a vector by a matrix and return the result
void vectorMatrixMultiplication(double *result, double *vec, double **matrix, int columnCountVector, int rowCountMatrix, int columnCountMatrix)
{
    double sum;
    if (columnCountVector == rowCountMatrix)
    {
        for (int i = 0; i < columnCountMatrix; i++)
        {
            sum = 0;
            #pragma omp parallel for reduction(+: sum) num_threads(2)
            for (int j = 0; j < columnCountVector; j++)
            {
                sum += vec[j] * matrix[j][i];
            }
            result[i] = sum;
        }
    }
}

int main()
{
    int nrow = 1500, ncol = 150000;
    double **m = (double**) calloc (nrow, sizeof(double*));
    double *v = (double*) calloc(nrow, sizeof(double));
    double *v2 = (double*) calloc(ncol, sizeof(double));

    for(int i = 0; i < nrow; i++) {
        m[i] = (double*) calloc(ncol + 1, sizeof(double));
    }

    vectorMatrixMultiplication(v2, v, m, nrow, nrow, ncol);
    cout << "ok\n";

    /*int nrow = 30, ncol = 150000;
    double **m = (double**) calloc (nrow, sizeof(double*));
    double *v = (double*) calloc(ncol, sizeof(double));

    for(int i = 0; i < nrow; i++) {
        m[i] = (double*) calloc(ncol + 1, sizeof(double));
    }

    double *result = matrixVectorMultiplication(m, v, nrow, ncol + 1, ncol);
    cout << "ok\n";*/
}