#!/bin/bash

g++ mlp.cpp -o seqMLP.exe -O3 -std=c++14

echo ""
echo "Sequencial:"
for i in {1..5}
do
    echo ""
    echo "Tentativa " $i
    time ./seqMLP.exe 512 0.1 0.001 1 < datasets/sampleNormalizedFonts.in
done

g++ mlp.cpp -o parMLP.exe -O3 -std=c++14 -fopenmp

echo ""
echo ""
echo "Paralelo com 2 threads:"
for i in {1..5}
do
    echo ""
    echo "Tentativa " $i
    time ./parMLP.exe 512 0.1 0.001 2 < datasets/sampleNormalizedFonts.in
done

echo ""
echo ""
echo "\n\nParalelo com 4 threads:"
for i in {1..5}
do
    echo ""
    echo "Tentativa " $i
    time ./parMLP.exe 512 0.1 0.001 4 < datasets/sampleNormalizedFonts.in
done

echo ""
echo ""
echo "Paralelo com 8 threads:"
for i in {1..5}
do
    echo ""
    echo "Tentativa " $i
    time ./parMLP.exe 512 0.1 0.001 8 < datasets/sampleNormalizedFonts.in
done