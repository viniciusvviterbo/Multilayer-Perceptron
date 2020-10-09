#!/bin/bash

g++ parMLP.cpp -o seqMLP -O3 -std=c++14
echo "Sequencial:"
for i in {1..5}
do
    echo "Tentativa " $i
    time ./seqMLP 1024 0.1 0.001 1 < datasets/sampleNormalizedFonts.in
done

g++ parMLP.cpp -o parMLP -O3 -std=c++14 -fopenmp

echo "Paralelo com 2 threads:"
for i in {1..5}
do
    echo "Tentativa " $i
    time ./parMLP 1024 0.1 0.001 2 < datasets/sampleNormalizedFonts.in
done

echo "Paralelo com 4 threads:"
for i in {1..5}
do
    echo "Tentativa " $i
    time ./parMLP 1024 0.1 0.001 4 < datasets/sampleNormalizedFonts.in
done