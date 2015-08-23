#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define N 200
#define LEARN_RATE 0.01
#define EPS 1e-11
double input[N], output[N];
void data(){
    srand(time(0));
    int i;
    double atom=M_PI_4/N;
    double x=-M_PI_4;
    for (i=0; i<N;++i){
        x+=atom;
        input[i]=x;
        output[i]=sin(x)+(rand()%200-100)*0.0001;
    }
}
double SQR(double x){return x*x;}
int main(void){
    double w1,w2,w3,w4;
    double y1,y2,y3,err,deltaW;
    double lastErr=1e10;
    int i,iter;
    w1=1.0;
    w2=-1.0;
    w3=0.01;
    w4=1.0;
    data();
    puts("Data:");
    for (i=0;i<N;++i){
        printf("%lf %lf\n",input[i],output[i]);
    }
    puts("Training...");
    for (iter=1;1;iter++){
        for (i=0;i<N;++i){
            y1=w1*input[i]+w2;
            y2=1.0/(1+exp(-y1));
            y3=w3*y2+w4;
            err=output[i]-y3;
            //compute w1
            deltaW=err*w3*y2*(1.0-y2)*input[i];
            w1+=deltaW*LEARN_RATE;
            //compute w2
            deltaW=err*w3*y2*(1.0-y2);
            w2+=deltaW*LEARN_RATE;
            //compute w3
            deltaW=err*y2;
            w3+=deltaW*LEARN_RATE;
            //compute w4
            deltaW=err;
            w4+=deltaW*LEARN_RATE;
        }
        err=0;
        for (i=0;i<N;++i){
            y1=w1*input[i]+w2;
            y2=1.0/(1+exp(-y1));
            y3=w3*y2+w4;
            err+=SQR(y3-output[i]);
        }
        err/=N;
        printf("Iter=%d, w1=%lf, w2=%lf, w3=%lf, w4=%lf, avgError=%.8lf\n",iter,w1,w2,w3,w4,err);
        if (lastErr-err<EPS){
            break;
        }
        lastErr=err;
    }
    puts("Training finished!\nResult:");
    printf("w1=%lf, w2=%lf, w3=%lf, w4=%lf, avgError=%.8lf\n",w1,w2,w3,w4,err);
    return 0;
}
