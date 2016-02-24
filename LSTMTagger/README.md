# LSTMTagger

### Introduction

This is only a simple demo of LSTM. The network architecture is shown below:

*Input* -> *Lookup table* -> *LSTM Block* ->*Softmax*

And loss function is cross entropy. The default version is *main.cpp*, while the version in *peephole.cpp* is a LSTM with peephole. 

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

Since this project is just a demo, don't hope it will get a state-of-the-art performance. This tagger use a character-level inference, which is much worse than sentence-level inference algorithm like Viterbi decoding.

### Reference

* Greff, Klaus, et al. *LSTM: A search space odyssey.* *arXiv preprint arXiv:1503.04069* (2015).
  
* Tai, Kai Sheng, Richard Socher, and Christopher D. Manning. *Improved semantic representations from tree-structured long short-term memory networks.* *arXiv preprint arXiv:1503.00075* (2015).
  
  ​