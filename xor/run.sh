#!/usr/bin/sh
echo "Building..."
make clean
make all

echo "Input training data size:"
./datagen > train_data

echo "Input test data size:"
./datagen > test_data

./main
