#ifndef LINEAR_ALGEBRA_H
#define LINEAR_ALGEBRA_H

#include <cstring>
template <int N>
struct VectorT{
    double d[N];
    VectorT(){
        memset(d,0,sizeof(d));
    }
    VectorT<N> operator+(const VectorT<N> v){
        VectorT<N> ret;
        for (int i=0;i<N;++i){
            ret.d[i]=d[i]+v.d[i];
        }
        return ret;
    }
};

template <int N,int M>
struct MatrixT{
    double d[N][M];
    MatrixT(){
        memset(d,0,sizeof(d));
    }
    VectorT<N> operator *(const VectorT<M> &v2){
        VectorT<N> v;
        for (int i=0;i<N;++i){
            for (int j=0;j<M;++j){
                v.d[i]+=d[i][j]*v2.d[j];
            }
        }
        return v;
    }
};

template <int N,int M>
VectorT<N+M> mergeVector(const VectorT<N> &v1, const VectorT<M> &v2){
    VectorT<N+M> v;
    for (int i=0;i<N;++i){
        v.d[i]=v1.d[i];
    }
    for (int j=0;j<M;++j){
        v.d[N+j]=v2.d[j];
    }
    return v;
}

template <int N>
VectorT<N+1> mergeBias(const VectorT<N> &v1){
    VectorT<N+1> v;
    for (int i=0;i<N;++i){
        v.d[i]=v1.d[i];
    }
    v.d[N]=1.0;
    return v;
}
#endif
