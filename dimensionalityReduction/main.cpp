#include <iostream>
#include <memory>
#include <cmath>
#include <fstream>

using std::cerr;
using std::endl;
using std::cout;
using std::ifstream;
double err_dump;
template <int N>
struct VectorT{
    double d[N];
    VectorT(){
        for (int i=0;i<N;++i) d[i]=0;
    }
    VectorT(const VectorT<N>& v){
        for (int i=0;i<N;++i) d[i]=v[i];
    }
    double & operator[](int i){
        if (i<0 || i>=N) {
            cerr << "VectorT: Index out of range." << endl; 
            return err_dump;
        }
        return d[i];
    }
    double operator[](int i) const{
        if (i<0 || i>=N) {cerr << "VectorT: Index out of range." << endl; return err_dump;}
        return d[i];
    }
    int length() const{
        return N;
    }
};

template <int N,int M>
struct MatrixT{
    double d[N][M];
    MatrixT(){
        for (int i=0;i<N;++i){
            for (int j=0;j<M;++j){
                d[i][j]=0;
            }
        }
    }
    VectorT<N> operator*(const VectorT<M>&v) const{
        VectorT<N> ret;
        for (int i=0;i<N;++i){
            for (int j=0;j<M;++j){
                ret[i]+=d[i][j]*v[j];
            }
        }
        return ret;
    }
};
template <int N, int M>
std::ostream& operator<<(std::ostream& out, const MatrixT<N,M>& m){
    out <<"[" <<endl;
    for (int i=0;i<N;++i){
        out <<"    [";
        for (int j=0;j<M;++j){
            out <<m.d[i][j] <<", ";
        }
        out <<"]," <<endl;
    }
    out <<"]" <<endl;
}

template <int N>
VectorT<N> sigmoid(const VectorT<N>& v){
    VectorT<N> ret;
    for (int i=0;i<N;++i){
        ret[i]=1/(1+exp(-v[i]));
    }
    return ret;
}

template <int N>
VectorT<N+1> mergeBias(const VectorT<N>& v){
    VectorT<N+1> ret;
    for (int i=0;i<N;++i){
        ret[i]=v[i];
    }
    ret[N]=1;
    return ret;
}

MatrixT<2,5> W1;
MatrixT<4,3> W2;
void initMatrix(){
    for (int i=0;i<2;++i){
        for (int j=0;j<5;++j){
            W1.d[i][j]=0.01;
        }
    }
    for (int i=0;i<4;++i){
        for (int j=0;j<3;++j){
            W2.d[i][j]=0.01;
        }
    }
}
double getError(VectorT<4> d_v){
    VectorT<4> o_v(W2*mergeBias(W1*mergeBias(d_v)));
    double err=0;
    for (int i=0;i<4;++i){
        err+=(o_v[i]-d_v[i])*(o_v[i]-d_v[i]);
    }
    return err;
}
double trainingItem(VectorT<4>d_v,int dim){
    const double LEARN_RATE = 0.05;
    VectorT<5> input(mergeBias(d_v));
    VectorT<3> h= mergeBias(W1*input);
    VectorT<4> output(W2*h);
    int N= 4;
    //back propagation
    double err;
    double delta;
    double errCount=0;
//    for (int j=0;j<2;++j){
    int j=dim;
        for (int k=0;k<5;++k){
            err=0.0;
            delta=0.0;
            for (int i=0;i<4;++i){
                err= d_v[i]-output[i];
                delta += err*W2.d[i][j];
                if (err>0) {delta*=err;} else {delta*= -err;}
            }
            delta*=LEARN_RATE*input[k];
            W1.d[j][k]+=delta;
        }
//    }

    for (int i=0;i<N;++i){
        err = d_v[i] - output[i];
        for (int j=0;j<3;++j){
            delta= LEARN_RATE*err*h[j];
            W2.d[i][j]+=delta;
        }
    }
}
void train(int dim){
    ifstream fin("input");
    int N;
    fin >> N;
    for (int i=0;i<N;++i){
        VectorT<4> d_v;
        fin >> d_v[0] >> d_v[1] >>d_v[2] >> d_v[3];
        trainingItem(d_v,dim);
    }
}
double getTotalError(){
    ifstream fin("input");
    int N;
    fin >> N;
    double result=0;
    for (int i=0;i<N;++i){
        VectorT<4> d_v;
        fin >> d_v[0] >> d_v[1] >>d_v[2] >> d_v[3];
        double tmp=getError(d_v);
        result+=tmp;
    }
    result /= N;
    return result;
}
void printRecover(){
    ifstream fin("input");
    int N;
    fin >> N;
    double result=0;
    for (int i=0;i<N;++i){
        VectorT<4> d_v;
        fin >> d_v[0] >> d_v[1] >>d_v[2] >> d_v[3];
        VectorT<4> o_v= W2*mergeBias(W1*mergeBias(d_v));
        cout << "Input=["<<d_v[0] <<", "<<d_v[1] <<", "<<d_v[2] <<", "<<d_v[3] <<"]" <<endl;
        cout << "Output=["<<o_v[0] <<", "<<o_v[1] <<", "<<o_v[2] <<", "<<o_v[3] <<"]" <<endl;
    }

}
int main(){
    cout << " ### Dimensionality Reduction Demo ###" <<endl;
    initMatrix();
    const double errThreshold=0.02;
    for (int Iter=0;Iter<10;Iter++){
        cout <<"=== Big Iter = " << Iter <<" ==="<<endl; 
        for (int i=0;i<100;++i){
            train(0);
            cout << " - Iter "<< i <<": error=" << getTotalError() <<endl;
        }
        for (int i=100;i<200;++i){
            train(1);
            cout << " - Iter "<< i <<": error=" << getTotalError() <<endl;
        }
        if (getTotalError()<errThreshold) break;
    }
    cout <<"=== Result ===" <<endl;
    printRecover();
    cout <<"W1="<<W1;
    cout <<"W2="<<W2;
    return 0;
}
