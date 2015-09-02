#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "linear_algebra.h"
const int MAX_INPUT=500;
const int HIDDEN_N=3;
const int OUTPUT_N=1;
const int INPUT_N=2;
using std::ifstream;
using std::cout;
using std::endl;
int inputCount;
VectorT<INPUT_N> input[MAX_INPUT];
VectorT<OUTPUT_N> teacher[MAX_INPUT];

MatrixT<HIDDEN_N, INPUT_N+1> Win; //weight for input
MatrixT<HIDDEN_N, HIDDEN_N> Wxx; //internal weight for hidden layer
MatrixT<OUTPUT_N, HIDDEN_N+1> Wout; //output weight

VectorT<HIDDEN_N> hiddenValue[MAX_INPUT];
VectorT<HIDDEN_N> beforeHidden[MAX_INPUT];
VectorT<OUTPUT_N> outputValue[MAX_INPUT]; // hidden and output layer -> unfold
double hiddenDelta[MAX_INPUT][INPUT_N +HIDDEN_N+1];
double outputDelta[MAX_INPUT];

template <int N>
VectorT<N> Tanh(const VectorT<N> & v){
    VectorT<N> ret;
    for (int i=0;i<N;++i){
        ret.d[i]=tanh(v.d[i]);
    }
    return ret;
}
void readInputFile(){
    ifstream fin("input");
    fin >> inputCount;
    for (int i=0;i<inputCount; ++i){
        fin >> input[i].d[0] >> input[i].d[1] >>teacher[i].d[0];
    }
    fin.close();
}
void computeNetwork(){
    //start layer
    beforeHidden[0]=Win*mergeBias(input[0]);
    hiddenValue[0]=Tanh(beforeHidden[0]);
    outputValue[0]=Wout*mergeBias(hiddenValue[0]);

    //later layers
    for (int i=1;i<inputCount;++i){
        beforeHidden[i]= Win*mergeBias(input[i]) + Wxx * hiddenValue[i-1];
        hiddenValue[i]=Tanh(beforeHidden[i]);
        outputValue[i]=Wout*mergeBias(hiddenValue[i]);
    }
}
double getError(){
    double error2=0.0;
    for (int i=0;i<inputCount;++i){
        double tmp=outputValue[i].d[0]-teacher[i].d[0];
        error2+=tmp*tmp;
    }
    return error2;
}
void training(double LEARN_RATE){
    computeNetwork();
    //compute delta on unit
    int T= inputCount-1;
    outputDelta[T] = teacher[T].d[0] - outputValue[T].d[0];
    outputDelta[T] *=1;
    for (int j=0;j<5;++j){
        hiddenDelta[T][j] = outputDelta[T]*Wout.d[0][j];
        hiddenDelta[T][j] *=(1+hiddenValue[T].d[j])*(1-hiddenValue[T].d[j]);
    }

    for (int i=inputCount-2 ; i>=0 ;--i){
        outputDelta[i] = teacher[i].d[0] - outputValue[i].d[0];
        outputDelta[i] *=1;
        for (int j=0;j<5;++j){
            hiddenDelta[i][j] = outputDelta[i]*Wout.d[0][j];
            hiddenDelta[i][j] *= (1+hiddenValue[T].d[j]) * (1-hiddenValue[T].d[j]);
        }
    }
    //compute weight delta
    //Wout
    double delta=0.0;
    for (int i=0;i<HIDDEN_N;++i){
        delta=0.0;
        for (int j=0;j<inputCount;++j){
            delta+=outputDelta[j]*hiddenValue[j].d[i];
        }
        delta/=inputCount;
        Wout.d[0][i]+=LEARN_RATE*delta;
    }
    //Wout bias
    delta=0.0;
    for (int j=0;j<inputCount;++j){
        delta+=outputDelta[j];
    }
    delta/=inputCount;
    Wout.d[0][HIDDEN_N]+=LEARN_RATE*delta;

    //Win
    for (int i=0;i<INPUT_N;++i){
        for (int j=0;j<HIDDEN_N;++j){
            delta=0.0;
            for (int k=0;k<inputCount;++k){
                delta+=input[k].d[i]*hiddenDelta[k][j];
            }
            delta/=inputCount;
            Win.d[j][i]+=LEARN_RATE*delta;
       
        }
    }
    //Win bias
    for (int j=0;j<HIDDEN_N;++j){
        delta=0.0;
        for (int k=0;k<inputCount;++k){
            delta+=hiddenDelta[k][j];
        }
        delta/=inputCount;
        Win.d[j][INPUT_N]+=LEARN_RATE*delta;
    }
    //Wxx
    for (int i=0;i<HIDDEN_N;++i){
        for (int j=0;j<HIDDEN_N;++j){
            delta=0.0;
            for (int k=0;k<inputCount;++k){
                delta+=hiddenValue[k-1].d[i]*hiddenDelta[k][j];
            }
            delta/=inputCount;
            Wxx.d[j][i]+=LEARN_RATE*delta;
        }
    }
}
template <int N,int M>
void initMatrix(MatrixT<N,M>& m){
    for (int i=0;i<N;++i){
        for (int j=0;j<M;++j){
            double t=rand()%10000;
            t/=1000;
            m.d[i][j]=t-5;
        }
    }
}

template <int N,int M>
void printMatrix(std::ostream& out,MatrixT<N,M>& m){
    out <<"[" <<endl;
    for (int i=0;i<N;++i){
        out << "[";
        for (int j=0;j<M;++j){
            out << m.d[i][j] <<", ";
        }
        out <<"]" <<endl;
    }
    out <<"]"<<endl;
}
int main(int argc, char ** argv){
    readInputFile();
    srand(time(0));
    initMatrix(Win);
    initMatrix(Wout);
    initMatrix(Wxx);
    cout <<"=== Training Started ===" <<endl;
    for (int i=0;i<=30000;++i){
        training(0.15);
        if (i%200) continue;
        cout << "Iter " << i <<": error=" <<getError() << endl;
    }
    cout <<"=== Matrix Got ===" <<endl;
    cout << "Win=" ;
    printMatrix(cout,Win);
    cout << "Wxx=" ;
    printMatrix(cout,Wxx);
    cout << "Wout=" ;
    printMatrix(cout,Wout);
    cout <<"=== Compute result ===" <<endl;
    computeNetwork();
    for (int i=0;i<inputCount;++i){
        cout <<"x1="<<input[i].d[0] <<" x2="<< input[i].d[1] 
            <<" y="<<teacher[i].d[0] <<"  Model="<< outputValue[i].d[0]<<endl;
    }
    return 0;
}
