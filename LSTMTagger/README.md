# LSTMTagger

### Introduction

This is only a simple demo of LSTM. The network architecture is shown below:

*Input* -> *Lookup table* -> *LSTM Block* ->*Softmax*

And loss function is cross entropy.

### Data Format

All data should be in below text format:

* All sequences contains some pairs of token and tag.
* A space(the 32nd char in ASCII) between the token and the tag.
* An empty line split two sequences.

The File *demo.txt* is a demo of the data.

### Configure

All hyperparameters are in config.h.

*OUTPUT_SIZE* should be the tag count of data plus one. The first position of the output layer is not used.

### Other

Since this project is just a demo, don't hope it will get a state-of-the-art performence. 