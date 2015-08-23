BPNet4Xor
===
A demo of back propogation algorithm, coding in C.

##Layer Introduction##
3-Layer Back Propogation Net to get Sin(x)

* Layer1(Linear):    y1 = w1\*x+w2
* Layer2(Nonlinear): y2 = (1+Exp(-y1))^(-1)
* Layer3(Linear):    y3 = w3\*x+w4

##Data Generation##
x1^x2
a noise per 200 data

##Execution##
./run.sh

the shell script will run make and data generator
