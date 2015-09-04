RNN: Backpropagation through time
======
A demo of BPTT.
###Intro###
Hidden layer will send some signals to itself. 

It has only 3 layer.
* Nonlinear function of hidden layer is tanh.
* Nonlinear function of output layer is 1/(1+exp(-x)).

The demo can only fit small data since it is without batch operation.

###initMatrix() is the key to success!!!!###
The initial weight is random value between -1 and 1.

