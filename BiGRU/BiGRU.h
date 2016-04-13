#ifndef __BIGRU_H
#define __BIGRU_H
#include "GRU.h"
#include <map>
#include <string>
#include <vector>
const int EMBED_SIZE = 100;
const int HIDDEN_SIZE = 100;
const int FC1_SIZE = 48;
const int TAG_SIZE = 48;
const int BATCH_SIZE = 1;
const double LEARN_RATE = 0.01;
const double WEIGHT_DECAY = 0.0001;
const int ITER_COUNT = 100;
const double MARGIN = 0.0;
const extern char* START_TAG;
struct NetworkParam {
	std::map<std::string, dvec<EMBED_SIZE> *> lookUpTable;
	GRUParam<EMBED_SIZE, HIDDEN_SIZE> gru1, gru2;
	dmat<TAG_SIZE, HIDDEN_SIZE * 2> W1;
	dvec<TAG_SIZE> B1;
	void init();
};

struct TokenStatus {
	dvec<EMBED_SIZE>* tokenEmbed;
	GRUStatus<EMBED_SIZE, HIDDEN_SIZE> gru1, gru2;
	dvec<HIDDEN_SIZE * 2> concat;
	dvec<TAG_SIZE> fc1;
};
struct TokenGradient {
	GRUGradient<EMBED_SIZE, HIDDEN_SIZE> gru1, gru2;
	dmat<TAG_SIZE, HIDDEN_SIZE * 2> W1;
	dvec<TAG_SIZE> B1;
};

struct SentenceStatus {
	std::vector<TokenStatus *> items;
	int * predictLabel = nullptr;
	void clear() {
		for (int i = 0; i < items.size(); ++i) {
			delete items[i];
		}
		items.clear();
		if (predictLabel != nullptr) {
			delete[] predictLabel;
			predictLabel = nullptr;
		}
	}
	size_t size() const {
		return items.size();
	}
};

struct SentenceGradient {
	std::vector<TokenGradient *> items;
	SentenceStatus *status;
	void clear() {
		for (int i = 0; i < items.size(); ++i) {
			delete items[i];
		}
		items.clear();
		status = nullptr;
	}
};

#endif
