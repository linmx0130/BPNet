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
        if (i<0 || i>=N) {cerr << "VectorT: Index out of range." << endl; return err_dump;}
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

MatrixT<2,3> W1;
MatrixT<1,3> W2;
void initMatrix(){
    W1.d[0][0]=0.01;
    W1.d[0][1]=0.01;
    W1.d[0][2]=0.01;
    W1.d[1][0]=0.01;
    W1.d[1][1]=0.01;
    W1.d[1][3]=0.01;
    W2.d[0][0]=0.01;
    W2.d[0][1]=0.01;
    W2.d[0][2]=0.01;
}
double getAns(double x1,double x2){
    VectorT<3> x;
    x[0]=x1;x[1]=x2;x[2]=1;
    VectorT<2> y1(W1*x);
    VectorT<3> y2(mergeBias(sigmoid(y1)));
    double ans=(W2*y2)[0];
    return ans;
}
double trainingItem(double x1,double x2,double target){
    const double LEARN_RATE=0.05;
    //calculate ans
    VectorT<3> x;
    x[0]=x1;x[1]=x2;x[2]=1;
    VectorT<2> y1(W1*x);
    VectorT<3> y2(mergeBias(sigmoid(y1)));
    double ans=(W2*y2)[0];
    
    //modify parameters
    double delta;
    double err=target-ans;
    
    //w1
    for (int i=0;i<2;++i){
        for (int j=0;j<3;++j){
            delta=LEARN_RATE*err*W2.d[0][i]*y2[i]*(1.0-y2[i])*x[j];
            W1.d[i][j]+=delta;
        }
    }
    //w2
    for (int i=0;i<3;++i){
        delta=LEARN_RATE*err*y2[i];
        W2.d[0][i]+=delta;
    }

    return err;
}
void train(){
    ifstream fin("train_data");
    int N;
    fin >>N;
    for (int i=0;i<N;++i){
        int x1,x2,y;
        fin >>x1 >> x2 >> y;
        trainingItem(x1,x2,y);
    }
    fin.close();
}
double testModel(bool outputData){
    ifstream fin("test_data");
    int N;
    fin >>N;
    double errCount=0;
    for (int i=0;i<N;++i){
        int x1,x2,y;
        fin >>x1 >> x2 >> y;
        double t=getAns(x1,x2);
        errCount+= (t-y)*(t-y);
        if (outputData) cout <<"Target=" <<y<<" Model answer=" <<t <<endl;
    }
    fin.close();
    return errCount/N;
}
int main(){
    initMatrix();
    const double eps=1e-3;
    int counter=0;
    double lasterr=100;
    do {
        train();
        double err=testModel(false);
        cout << "Iter " << counter++ <<": error="<< err*100 <<"%" <<endl;;
        if (err<eps) break;
    }while (1);

    cout <<"W1= [" <<endl;
    for (int i=0;i<2;++i){
        for (int j=0;j<3;++j){
            cout <<W1.d[i][j] <<", ";
        }
        cout << endl;
    }
    cout <<"]"<<endl;
    
    cout <<"W2= [" <<endl;
    for (int i=0;i<1;++i){
        for (int j=0;j<3;++j){
            cout <<W2.d[i][j] <<", ";
        }
        cout << endl;
    }
    cout <<"]"<<endl;
    testModel(true);

    return 0;
}
