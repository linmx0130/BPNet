# BiGRU: a bidirectional GRU Tagger

### Introduction

This is only a simple demo of BiGRU. The network architecture is shown below:

*Input* -> *Lookup table* -> *BiGRU* -> *FC* -> *Softmax*

And loss function is cross entropy.

### Data Format

All data should be in below text format:

* All sequences contains some pairs of token and tag.
* A space(the 32nd char in ASCII) between the token and the tag.
* An empty line split two sequences.

The File *demo.txt* in LSTMTagger is a demo of the data.

  ​