#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#define N 10
double getDoubleRand(){
    return ((double )(rand()%10))/10;
}
int main(){
    srand(time(0));
    int DCOUNT = 5;
    printf("%d\n",DCOUNT);
    while (DCOUNT--)
    {
        printf("%d\n",N);
        double x1=0,x2=1,y=0;
        for (int i=0;i<N;++i){
            if (getDoubleRand()<0.7){
                x1=1;
            }else {
                x1=0;
            }
            if (x1==1){
            //    x2=getDoubleRand()+0.1;
                if (y == 1) y=0;
                else y = 1;
            }
            printf("%.1lf %.1lf %.1lf\n",x1,x2,y);
        }
    }
    return 0;
}
