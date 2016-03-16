#ifndef _CONV_SEN_H
#define _CONV_SEN_H
#include "cornernn_math.h"
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cstdlib>
#define WORD2VEC_INIT
using std::vector;
using std::string;
using namespace cornernn;
const int CHAR_EMBED_SIZE = 100;
const int KERNEL_3_COUNT = 100;
const int KERNEL_4_COUNT = 100;
const int KERNEL_5_COUNT = 100;
const double LAMBDA = 0.0001;
const double DROPOUT_RATE = 0.8;
const int KERNEL_COUNT = KERNEL_3_COUNT + KERNEL_4_COUNT + KERNEL_5_COUNT;
const int BATCH_SIZE = 40;
const double LEARN_RATE = 0.01;
template <int size>
using dvec = Vector<double, size>;
struct Sentence {
	vector<dvec<CHAR_EMBED_SIZE>*> words;
	vector<string> tokens;
	int tag;
};
struct Kernel {
	vector<dvec<CHAR_EMBED_SIZE>*> filters;
	Kernel(int size) {
		for (int i = 0; i < size; ++i) {
			filters.push_back(new dvec<CHAR_EMBED_SIZE>);
			filters[i]->initToRandom(sqrt(CHAR_EMBED_SIZE * size));
		}
	}
};

const char *NEG_DATA = "rt-polaritydata/rt-polarity.neg";
const char *POS_DATA = "rt-polaritydata/rt-polarity.pos";
const char *WORD_VEC = "word2vec_d100.txt";
const int TAG_COUNT = 2;
struct RunningStatus {
	Sentence *sentence;
	double **c;
	int *maxC;
	dvec<KERNEL_COUNT> z;
	dvec<TAG_COUNT> y;
	void init() {
		c = new double*[KERNEL_COUNT];
		maxC = new int[KERNEL_COUNT];
		for (int i = 0; i < KERNEL_COUNT; ++i) {
			c[i] = nullptr;
			maxC[i] = 0;
		}
	}
	void clean() {
		for (int i = 0; i < KERNEL_COUNT; ++i) {
			if (c[i] != nullptr) {
				delete[] c[i];
				c[i] = nullptr;
			}
		}
		delete[] c;
		delete[] maxC;
		sentence = nullptr;
	}
	RunningStatus() {
		init();
	}
	~RunningStatus() {
		clean();
	}
};
struct GradientStatus {
	RunningStatus *rs;
	dvec<TAG_COUNT> dy;
	dvec<KERNEL_COUNT> dz;
	std::map<string, dvec<CHAR_EMBED_SIZE>> dLookUp;
	void clean() {
		rs = nullptr;
		dy.initToZero();
		dz.initToZero();
		dLookUp.clear();
	}
};
#endif 