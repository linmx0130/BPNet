#include <ctime>
#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace std;
#define TTT 100007
double getRandDouble(){
    return ((double)(rand()%TTT))/TTT;
}
int main(int argc, char ** argv){
    int N;
    int t;
    ofstream fout(argv[1]);
    cout <<"Input numbers of true variable(1/2):";
    cin >> t;
    if (t<1 || t>2) return 1;
    cout <<"Input data size:";
    cin >> N;
    fout <<N <<endl;
    srand(time(0));
    for (int i=0;i<N;++i){
        double x1 = getRandDouble(), x2, x3=getRandDouble(),x4;
        x2=x1+(getRandDouble()-0.5)/300;
        if (t==1) x3=x2;
        x4=x3*2+(getRandDouble()-0.5)/300;
        x3=-x3;
        fout <<x1*3+0.001 << " " <<x2 << " " <<x3 <<" "<<x4 <<endl;
    }
    fout.close();
    return 0;
}
