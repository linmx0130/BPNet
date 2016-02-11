#include "cornernn_math.h"
#include <cstdio>
#include <vector>
#include <memory>
#include <ctime>
using namespace cornernn;
using std::unique_ptr;
namespace LeNet {
	template<int ROW, int COLUMN>
	using dmat = Matrix<double, ROW, COLUMN>;
	char *inputsDir = "testset/";
	const int DATA_COUNT = 3000;
	dmat<28, 28> inputs[DATA_COUNT];
	int targets[DATA_COUNT];
	const int CONV1_FCOUNT = 6;
	dmat<5, 5> conv1filters[CONV1_FCOUNT];
	const int CONV2_FCOUNT = 10;
	dmat<5, 5> conv2filters[CONV2_FCOUNT];
	const int FC2_SIZE = 50;
	const int FC1_SIZE = 4 * 4 * CONV2_FCOUNT;
	dmat< FC2_SIZE, FC1_SIZE> Wfc1;
	dmat< 10, FC2_SIZE > Wfc2;
	const double LEARNING_RATE = 0.01;
	const double SHRINK_RATE = 0.9999;

	void loadInputs() {
		char buf[10];
		for (int i = 0; i < DATA_COUNT; ++i) {
			std::string fileName(inputsDir);
			_itoa_s(i+1, buf, 10);
			fileName += buf;
			fileName += ".txt";
			FILE *fin = nullptr;
			fopen_s(&fin, fileName.c_str(), "r");
			for (int x = 0; x < 28; ++x) {
				for (int y = 0; y < 28; ++y) {
					double u;
					fscanf_s(fin, "%lf", &u);
					u = (u-128) / 128;
					inputs[i].d[x][y] = u;
				}
			}
			fscanf_s(fin, "%d", &targets[i]);
			fclose(fin);
		}
		printf("Loaded data.\n");
	}

	dmat<24, 24> convLayer1[CONV1_FCOUNT];
	dmat<12, 12> poolingLayer1[CONV1_FCOUNT];
	dmat<8, 8> convLayer2[CONV2_FCOUNT];
	dmat<4, 4> poolingLayer2[CONV2_FCOUNT];
	Vector<double, 4 * 4 * CONV2_FCOUNT> fcLayer1;
	Vector<double, FC2_SIZE> fcLayer2;
	Vector<double, 10> softmaxOutput;

	Vector<double, 10> error;
	Vector<double, FC1_SIZE> gradFcLayer1;
	Vector<double, FC2_SIZE> gradFcLayer2;
	dmat<4, 4> gradPoolingLayer2[CONV2_FCOUNT];
	dmat<8, 8> gradConvLayer2[CONV2_FCOUNT];
	dmat<12, 12> gradPoolingLayer1[CONV1_FCOUNT];
	dmat<24, 24> gradConvLayer1[CONV1_FCOUNT];
	bool trainItem(const dmat<28, 28>& input, int target) {
		
		//conv1
		for (int i = 0; i < CONV1_FCOUNT; ++i) {
			MMConv<double, 28, 5, 1>(input, conv1filters[i], convLayer1[i]);
		}
		//activation1
		for (int i = 0; i < CONV1_FCOUNT; ++i) {
			convLayer1[i].perform(cornernn::ReLU<double>);
		}
		//subsampling1
		for (int i = 0; i < CONV1_FCOUNT; ++i) {
			MeanPooling<double, 24, 2>(convLayer1[i], poolingLayer1[i]);
		}
		//conv2
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			dmat<8, 8> convOutput;
			convLayer2[i].initToZero();
			for (int j = 0; j < CONV1_FCOUNT; ++j) {
				MMConv<double, 12, 5, 1>(poolingLayer1[j], conv2filters[i], convOutput);
				convLayer2[i] += convOutput;
			}
		}
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			convLayer2[i].perform(cornernn::ReLU<double>);
		}
		//subsampling2
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			MeanPooling<double, 8, 2>(convLayer2[i], poolingLayer2[i]);
		}
		//convert to fc
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			for (int x = 0; x < 4; ++x) {
				for (int y = 0; y < 4; ++y) {
					fcLayer1.d[i * 16 + x * 4 + y] = poolingLayer2[i].d[x][y];
				}
			}
		}
		MVRightMultiply(Wfc1, fcLayer1, fcLayer2);
		fcLayer2.perform(cornernn::ReLU<double>);
		MVRightMultiply(Wfc2, fcLayer2, softmaxOutput);
		softmaxOutput.performSoftmax();
		
		int ans = 0;
		//get error before softmax
		for (int i = 0; i < 10; ++i) {
			error.d[i] = softmaxOutput.d[i];
			if (softmaxOutput.d[i] > softmaxOutput.d[ans]) {
				ans = i;
			}
			if (target == i) error.d[i] -= 1;
		}

		//fc2
		MVLeftMultiply(Wfc2, error, gradFcLayer2);
		for (int i = 0; i < gradFcLayer2.getLength(); ++i) {
			gradFcLayer2.d[i] = fcLayer2.d[i] == 0 ? 0 : gradFcLayer2.d[i];
		}
		//fc1
		MVLeftMultiply(Wfc1, gradFcLayer2, gradFcLayer1);
		//convert to pooling layer 2
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			for (int x = 0; x < 4; ++x) {
				for (int y = 0; y < 4; ++y) {
					gradPoolingLayer2[i].d[x][y] = gradFcLayer1.d[i * 16 + x * 4 + y];
				}
			}
		}
		//conv2
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			InvMeanPooling<double, 8, 2>(gradPoolingLayer2[i],gradConvLayer2[i]);
		}
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			for (int x = 0; x < gradConvLayer2[i].getRowSize(); ++x) {
				for (int y = 0; y < gradConvLayer2[i].getColumnSize(); ++y) {
					gradConvLayer2[i].d[x][y] = convLayer2[i].d[x][y] == 0 ? 0 : gradConvLayer2[i].d[x][y];
				}
			}
		}
		//pooling layer 1
		for (int i = 0; i < CONV1_FCOUNT; ++i) {
			dmat<12, 12> gradTemp;
			gradPoolingLayer1[i].initToZero();
			for (int j = 0; j < CONV2_FCOUNT; ++j) {
				MMInvConv<double, 12, 5, 1>(gradConvLayer2[j], conv2filters[j], gradTemp);
				gradPoolingLayer1[i] += gradTemp;
			}
		}
		//conv1
		for (int i = 0; i < CONV1_FCOUNT; ++i) {
			InvMeanPooling<double, 24, 2>(gradPoolingLayer1[i], gradConvLayer1[i]);
		}
		for (int i = 0; i < CONV1_FCOUNT; ++i) {
			for (int x = 0; x < gradConvLayer1[i].getRowSize(); ++x) {
				for (int y = 0; y < gradConvLayer1[i].getColumnSize(); ++y) {
					gradConvLayer1[i].d[x][y] = convLayer1[i].d[x][y] == 0 ? 0 : gradConvLayer1[i].d[x][y];
				}
			}
		}
		//update fc2
		for (int i = 0; i < Wfc2.getRowSize(); ++i) {
			for (int j = 0; j < Wfc2.getColumnSize(); ++j) {
				Wfc2.d[i][j] -= LEARNING_RATE* error.d[i] * fcLayer2.d[j];
			}
		}
		//update fc1
		for (int i = 0; i < Wfc1.getRowSize(); ++i) {
			for (int j = 0; j < Wfc1.getColumnSize(); ++j) {
				Wfc1.d[i][j] -= LEARNING_RATE* gradFcLayer2.d[i] * fcLayer1.d[j];
			}
		}
		//update conv2
		dmat<5, 5> dConv;
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			for (int j = 0; j < CONV1_FCOUNT; ++j) {
				MMConv<double, 12, 8, 1>(poolingLayer1[j], gradConvLayer2[i], dConv);
				dConv *= LEARNING_RATE/CONV1_FCOUNT;
				conv2filters[i] -= dConv;
			}
		}

		for (int i = 0; i < CONV1_FCOUNT; ++i) {
			MMConv<double, 28, 24, 1>(input, gradConvLayer1[i], dConv);
			dConv *= LEARNING_RATE;
			conv1filters[i] -= dConv;
		}
		return ans == target;
	}

	int feedFoward(const dmat<28, 28>& input) {
		auto ReLU = [](double x) { return x > 0 ? x : 0; };
		//conv1
		for (int i = 0; i < CONV1_FCOUNT; ++i) {
			MMConv<double, 28, 5, 1>(input, conv1filters[i], convLayer1[i]);
		}
		//activation1
		for (int i = 0; i < CONV1_FCOUNT; ++i) {
			convLayer1[i].perform(ReLU);
		}
		//subsampling1
		for (int i = 0; i < CONV1_FCOUNT; ++i) {
			MeanPooling<double, 24, 2>(convLayer1[i], poolingLayer1[i]);
		}
		//conv2
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			dmat<8, 8> convOutput;
			convLayer2[i].initToZero();
			for (int j = 0; j < CONV1_FCOUNT; ++j) {
				MMConv<double, 12, 5, 1>(poolingLayer1[j], conv2filters[i], convOutput);
				convLayer2[i] += convOutput;
			}
		}
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			convLayer2[i].perform(ReLU);
		}
		//subsampling2
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			MeanPooling<double, 8, 2>(convLayer2[i], poolingLayer2[i]);
		}
		//convert to fc
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			for (int x = 0; x < 4; ++x) {
				for (int y = 0; y < 4; ++y) {
					fcLayer1.d[i * 16 + x * 4 + y] = poolingLayer2[i].d[x][y];
				}
			}
		}
		MVRightMultiply(Wfc1, fcLayer1, fcLayer2);
		fcLayer2.perform(ReLU);
		MVRightMultiply(Wfc2, fcLayer2, softmaxOutput);
		softmaxOutput.performSoftmax();

		int ans = 0;
		//get error before softmax
		for (int i = 0; i < 10; ++i) {
			error.d[i] = softmaxOutput.d[i];
			if (softmaxOutput.d[i] > softmaxOutput.d[ans]) {
				ans = i;
			}
		}
		return ans;
	}
	int main() {
		loadInputs();
		srand(time(0));
		Wfc1.initToRandom(sqrt(Wfc1.getColumnSize()));
		Wfc2.initToRandom(sqrt(Wfc2.getColumnSize()));
		for (int i = 0; i < CONV1_FCOUNT; ++i) {
			conv1filters[i].initToRandom(conv1filters[i].getColumnSize());
		}
		for (int i = 0; i < CONV2_FCOUNT; ++i) {
			conv2filters[i].initToRandom(conv2filters[i].getColumnSize()*CONV1_FCOUNT);
		}
		for (int loop = 0; loop < 50; ++loop) {
			int count = 0; 
			int tcount = 0;
			printf("Iter %d:\n",loop);
			for (int i = 0; i < DATA_COUNT-400 ; ++i) {
				count += trainItem(inputs[i], targets[i]) ?1:0;
				if (i % 100 == 99) {
					printf("   %d~%d, %d/100\n", i-99,i,count );
					tcount += count;
					count = 0;
				}
				if (isnan(softmaxOutput.d[0])) {
					printf("NAN ERROR!");
					while (1);
				}
			}
			printf(" Total: %d/%d -> %lf\n", tcount, DATA_COUNT-400, (double)tcount/(DATA_COUNT - 400));
			tcount = 0;
			count = 0;
			for (int i = DATA_COUNT - 400; i < DATA_COUNT; ++i) {
				count += feedFoward(inputs[i]) == targets[i];
			}
			printf(" Test: %d/%d ->%lf\n", count, 400, (double) count / 400);

			Wfc1 *= SHRINK_RATE;
			Wfc2 *= SHRINK_RATE;
			for (int i = 0; i < CONV1_FCOUNT; ++i) {
				conv1filters[i] *= SHRINK_RATE;
			}
			for (int i = 0; i < CONV2_FCOUNT; ++i) {
				conv2filters[i] *= SHRINK_RATE;
			}
		}
		while (1);
		return 0;
	}
};