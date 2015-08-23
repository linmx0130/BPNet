#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
int N;
int main(){
    scanf("%d",&N);
    printf("%d\n",N);
    srand(time(0));
    for (int i=0;i<N;++i){
        int x1=rand()%2;
        int x2=rand()%2;
        int y=x1^x2;
        if (N%200==1) y=!y;
        printf("%d %d %d\n",x1,x2,y);
    }
    return 0;
}
