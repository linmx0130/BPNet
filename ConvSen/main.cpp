#include "ConvSen.h"

std::map<string, dvec<100>* > lookUp;
vector<Sentence*> sentences;
vector<Kernel*> kernels;
Matrix<double, TAG_COUNT, KERNEL_COUNT> W1;
dvec<TAG_COUNT> B1;
RunningStatus runningStatus[BATCH_SIZE];
GradientStatus gradientStatus[BATCH_SIZE];
void loadData(const char* filename, int tag) {
	std::ifstream fin(filename);
	char buf[2048];
	dvec<100> *noneVec = new dvec<100>;
	noneVec->initToZero();
	lookUp["~NOWORD~"] = noneVec;
	while (fin.getline(buf,2048) ) {
		Sentence * s = new Sentence();
		s->tag = tag;
		std::istringstream sp(buf);
		string word;
		while (sp >> word) {
			dvec<100> * p = lookUp[word];
			if (p== nullptr) {
				dvec<100> *wv = new dvec<100>();
				wv->initToRandom(4);
				p=lookUp[word] = wv;
			}
			s->words.push_back(p);
			s->tokens.push_back(word);
		}
		while (s->words.size() < 5) {
			s->words.push_back(noneVec);
			s->tokens.push_back("~NOWORD~");
		}
		sentences.push_back(s);
	}
}
void initWeights() {
	for (int i = 0; i < KERNEL_3_COUNT; ++i) {
		kernels.push_back(new Kernel(3));
	}
	for (int i = 0; i < KERNEL_4_COUNT; ++i) {
		kernels.push_back(new Kernel(4));
	}
	for (int i = 0; i < KERNEL_5_COUNT; ++i) {
		kernels.push_back(new Kernel(5));
	}
	W1.initToRandom(sqrt(KERNEL_COUNT));
	B1.initToRandom(sqrt(KERNEL_COUNT));
}

void feedForward(Sentence *s, RunningStatus* rs) {
	rs->sentence = s;
	for (int k = 0; k < KERNEL_COUNT; ++k) {
		int l = s->words.size() - kernels[k]->filters.size() + 1;
		rs->c[k] = new double[l];
		for (int i = 0; i < l; ++i) {
			rs->c[k][i] = 0;
			for (int j = 0; j < kernels[k]->filters.size(); ++j) {
				rs->c[k][i] += VVEWProductSum(*s->words[i + j], *kernels[k]->filters[j]);
			}
			if (rs->c[k][i] < 0) rs->c[k][i] = 0;//relu
		}
		for (int i = 0; i < l; ++i) {
			if (rs->c[k][i] > rs->c[k][rs->maxC[k]]) {
				rs->maxC[k] = i;
			}
		}
		rs->z.d[k] = rs->c[k][rs->maxC[k]];
	}
	MVRightMultiply(W1, rs->z, rs->y);
	rs->y += B1;
	rs->y.performSoftmax();
}

void trainBack(RunningStatus *rs, GradientStatus* g) {
	g->rs = rs;
	for (int i = 0; i < TAG_COUNT; ++i) {
		g->dy.d[i] = rs->y.d[i];
	}
	g->dy.d[rs->sentence->tag] -= 1;

	MVLeftMultiply(W1, g->dy, g->dz);
	for (int k = 0; k < g->dz.getLength(); ++k) {
		if (rs->z.d[k] == 0) g->dz.d[k] = 0;
	}
	for (int k = 0; k < KERNEL_COUNT; ++k) {
		int maxC = rs->maxC[k];
		for (int i = 0; i < kernels[k]->filters.size(); ++i) {
			dvec<100> * v= &g->dLookUp[rs->sentence->tokens[maxC+i]];
			for (int j = 0; j < CHAR_EMBED_SIZE; ++j) {
				v->d[j] += g->dz.d[k] * kernels[k]->filters[i]->d[j];
			}
		}
	}
}
void updateParameters(GradientStatus* g) {
	Matrix <double, TAG_COUNT, KERNEL_COUNT> dW1;
	VVOuterProduct(g->dy, g->rs->z, dW1); dW1 *= LEARN_RATE;
	W1 -= dW1;
	dvec <TAG_COUNT> dB1(g->dy); dB1 *= LEARN_RATE;
	B1 -= dB1;
	for (int k = 0; k < KERNEL_COUNT; ++k) {
		int maxC = g->rs->maxC[k];
		for (int i = 0; i < kernels[k]->filters.size(); ++i) {
			dvec<100> *vi = g->rs->sentence->words[maxC + i];
			for (int j = 0; j < CHAR_EMBED_SIZE; ++j) {
				kernels[k]->filters[i]->d[j] -= LEARN_RATE* g->dz.d[k] * vi->d[j];
			}
		}
	}
}
void loadWordVector() {
	std::ifstream fin(WORD_VEC);
	int N, M;
	fin >> N >> M;
	dvec<100> tmp;
	string buf;
	int count = 0;
	for (int i = 0; i < N; ++i) {
		fin >> buf;
		for (int j = 0; j < M; ++j) {
			fin >>tmp.d[j];
		}
		if (lookUp.find(buf) != lookUp.end()) {
			(*lookUp[buf]) = tmp;
			count++;
		}
	}
	std::cout << "Load " << count << " words form word2vec." << std::endl;
}
void loadData() {
	loadData(NEG_DATA, 0);
	loadData(POS_DATA, 1);
	int N = sentences.size();
	for (int i = 0; i < N; ++i) {
		int a = rand() % N;
		int b = rand() % N;
		Sentence *k = sentences[a];
		sentences[a] = sentences[b];
		sentences[b] = k;
	}
	/*
	* if you want to use word2vec product to get better performence
	* use follow code!
	*/
#ifdef WORD2VEC_INIT
	loadWordVector();
#endif
}
bool isCorrect(const RunningStatus & rs) {
	int ans = 0;
	for (int i = 0; i < TAG_COUNT; ++i) {
		if (rs.y.d[i]>rs.y.d[ans]) ans = i;
	}
	return ans == rs.sentence->tag;
}
void regularization() {
	W1 *= 1 - LAMBDA;
	for (int i = 0; i < kernels.size(); ++i) {
		for (int j = 0; j < kernels[i]->filters.size(); ++j) {
			(*kernels[i]->filters[j]) *= 1 - LAMBDA;
		}
	}
}
int main() {
	srand(time(0));
	initWeights();
	loadData();
	int dataCount = sentences.size();
	int trainCount = dataCount*0.9;
	int testCount = dataCount - trainCount;
	std::cout << "Total valid data count = " << sentences.size() << std::endl;
	std::cout << "Train count = " << trainCount << std::endl;
	std::cout << "Test count = " << testCount << std::endl;
	for (int iter = 0; iter < 100; ++iter) {
		std::cout << "ITER " << iter << std::endl;
		int ii = 0;
		int totalCorrect = 0;
		int correct = 0;
		for (int i = 0; i < trainCount; ++i, ++ii) {
			runningStatus[ii].init();
			feedForward(sentences[i], &runningStatus[ii]);
			trainBack(&runningStatus[ii], &gradientStatus[ii]);
			if (isCorrect(runningStatus[ii])) {
				correct++;
			}
			if (ii == BATCH_SIZE - 1) {
				for (int j = 0; j < BATCH_SIZE; ++j) {
					updateParameters(&gradientStatus[j]);
					runningStatus[j].clean();
					gradientStatus[j].clean();
				}
				regularization();
				ii = -1;
				totalCorrect += correct;
				correct = 0;
			}
		}
		if (ii != 0) {
			for (int j = 0; j < ii; ++j) {
				updateParameters(&gradientStatus[j]);
				runningStatus->clean();
				runningStatus->init();
				gradientStatus->clean();
			}
			totalCorrect += correct;
		}
		std::cout << "ITER " << iter << ": train total correct rate = " << (double)totalCorrect / trainCount << std::endl;
		correct = 0;
		for (int i = trainCount; i < dataCount; ++i) {
			runningStatus[0].init();
			feedForward(sentences[i], &runningStatus[0]);
			if (isCorrect(runningStatus[0])) {
				correct++;
			}
			runningStatus[0].clean();
		}
		std::cout << "ITER " << iter << ": test correct rate = " << (double)correct / testCount << std::endl;
	}
	return 0;
}