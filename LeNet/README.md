# LeNet: Convolution Neural Network For Digits Recognition

A C++ implementation of the famous LeNet-5 with some simplifications.

## Introduction

The architech is similar to LeNet-5. However, this implementation use *ReLU* to replace *tanh* as the activation function.  The criteria is *softmax* function and the loss function is cross entropy.

The input data should be 28*28 integers in [0, 255] with a target label.

Maybe you need a C++14 compiler to build the program.

## Reference

LeCun, Yann, et al. "Gradient-based learning applied to document recognition." *Proceedings of the IEEE* 86.11 (1998): 2278-2324.