#include <cstdio>
#include <cstdlib>
#include <ctime>
using namespace std;
const int N = 100;
const int M = 20;
void genData(){
	FILE *fout;
	fopen_s(&fout, "data.in", "w");
	srand((unsigned int) time(0));
	fprintf(fout, "%d %d\n", N, M);
	for (int i = 0; i < N; ++i) {
		int c = 0;
		for (int j = 0; j < M; ++j) {
			int a = rand() % 2, b = rand() % 2;
			c = c ^a ;
			fprintf(fout, "%d %d %d\n", a, b, c);
		}
	}
	fclose(fout);
}