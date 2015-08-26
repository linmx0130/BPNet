BPNet4DimensionalityReduction
===
A demo of back propogation algorithm, coding in C++.

##Layer Introduction##
2-Layer Back Propogation Net to do linear dimensionality reduction.

* Layer1(Linear):    y1 = w1\*x+b1
* Layer2(Linear):    y2 = w2\*y1+b2

Minimize the difference of x and y2.
y1 is the answer exactly.

##Data Generation##
In datagen.cpp
Support to create information of 1 or 2 variable(s)

##Run##
./run.sh

