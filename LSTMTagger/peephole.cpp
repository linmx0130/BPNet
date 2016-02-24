#include "cornernn_math.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
using namespace cornernn;
using std::string;
template <int LEN>
using dvec = Vector<double, LEN>;
template <int ROW, int COLUMN>
using dmat = Matrix<double, ROW, COLUMN>;

const char *TRAIN_FILE = "train.utf8";
const char *TEST_FILE = "test.utf8";
const char *OUTPUT_FILE_NAME_PATTERN = "output_%d.utf8";
const int EMBEDDING_SIZE = 50;
const int HIDDEN_SIZE = 120;
const int OUTPUT_SIZE = 5;
const int ITER_COUNT = 10;
const double LEARN_RATE = 0.02;
const double SHRINK_RATE = 0.9995;
struct LSTMTaggerModel {
	//loopup table
	std::map<string, dvec<EMBEDDING_SIZE>> lookupTable;
	//LSTM parameters
	dmat<HIDDEN_SIZE, EMBEDDING_SIZE> Wi, Wf, Wo, Wu;
	dmat<HIDDEN_SIZE, HIDDEN_SIZE> Ui, Uf, Uo, Uu;
	dvec<HIDDEN_SIZE> Pi, Pf, Po;
	dvec<HIDDEN_SIZE> Bi, Bf, Bo, Bu;
	dmat<OUTPUT_SIZE, HIDDEN_SIZE> Wfc;
	
	void init() {
		Wi.initToRandom(sqrt(EMBEDDING_SIZE)*3);
		Wf.initToRandom(sqrt(EMBEDDING_SIZE)*3);
		Wo.initToRandom(sqrt(EMBEDDING_SIZE)*3);
		Wu.initToRandom(sqrt(EMBEDDING_SIZE)*3);
		Ui.initToRandom(sqrt(HIDDEN_SIZE)*3);
		Uf.initToRandom(sqrt(HIDDEN_SIZE)*3);
		Uo.initToRandom(sqrt(HIDDEN_SIZE)*3);
		Uu.initToRandom(sqrt(HIDDEN_SIZE)*3);
		Pi.initToRandom(sqrt(HIDDEN_SIZE)*3);
		Pf.initToRandom(sqrt(HIDDEN_SIZE)*3);
		Po.initToRandom(sqrt(HIDDEN_SIZE)*3);
		Bi.initToZero();
		Bf.initToZero();
		Bo.initToZero();
		Bu.initToZero();
		Wfc.initToRandom(sqrt(HIDDEN_SIZE));
	}
};
struct Delta {
	dvec<EMBEDDING_SIZE> input;
	dmat<HIDDEN_SIZE, EMBEDDING_SIZE> Wi, Wf, Wo, Wu;
	dmat<HIDDEN_SIZE, HIDDEN_SIZE> Ui, Uf, Uo, Uu;
	dvec<HIDDEN_SIZE> Pi, Pf, Po;
	dvec<HIDDEN_SIZE> Bi, Bf, Bo, Bu;
	dmat<OUTPUT_SIZE, HIDDEN_SIZE> Wfc;
};
struct LSTMState {
	dvec<EMBEDDING_SIZE> *input;
	dvec<HIDDEN_SIZE> cell, cellTanh, hidden, Gi, Go, Gu, Gf;
	dvec<OUTPUT_SIZE> output;
	LSTMState* lastState ;
	LSTMTaggerModel *model;

	LSTMState(LSTMTaggerModel *model) :input(nullptr), lastState(nullptr), model(model){
	}
	void init() {
		input = nullptr;
		cell.initToZero(); hidden.initToZero();
		lastState = nullptr;
	}
	void feedForward(LSTMState *lastState, const string& inputStr) {
		this->lastState = lastState;
		//get input char embedding
		input = &model->lookupTable[inputStr];
		
		//lstm
		dvec<HIDDEN_SIZE> tmp;
		
		MVRightMultiply(model->Wi, *input, tmp);
		MVRightMultiply(model->Ui, lastState->hidden, Gi);
		Gi += tmp; 
		tmp = lastState->cell; tmp *= model->Pi;
		Gi += tmp; Gi += model->Bi;  Gi.performSigmoid();
		
		MVRightMultiply(model->Wf, *input, tmp);
		MVRightMultiply(model->Uf, lastState->hidden, Gf);
		Gf += tmp;
		tmp = lastState->cell; tmp *= model->Pf;
		Gf += tmp; Gf += model->Bf; Gf.performSigmoid();

		MVRightMultiply(model->Wo, *input, tmp);
		MVRightMultiply(model->Uo, lastState->hidden, Go);
		Go += tmp;
		tmp = lastState->cell; tmp *= model->Po;
		Go += tmp; Go += model->Bo; Go.performSigmoid();

		MVRightMultiply(model->Wu, *input, tmp);
		MVRightMultiply(model->Uu, lastState->hidden, Gu);
		Gu += tmp; Gu += model->Bu; Gu.performTanh();

		cell = lastState->cell; cell *= Gf;
		tmp = Gi; tmp *= Gu;
		cell += tmp;

		cellTanh = cell; cellTanh.performTanh();
		hidden = cellTanh; hidden *= Go;

		//output to fc layer
		MVRightMultiply(model->Wfc, hidden, output);
		output.performSoftmax();
	}
	int getAnswer() const {
		int ans = 0;
		for (int i = 0; i < OUTPUT_SIZE; ++i) {
			if (output.d[i] > output.d[ans]) {
				ans = i;
			}
		}
		return ans;
	}
	void backProp(int target, Delta & d) {
		dvec<OUTPUT_SIZE> dErr;
		//derive loss function at softmax
		int ans = 0;
		for (int i = 0; i < OUTPUT_SIZE; ++i) {
			dErr.d[i] = output.d[i];
			if (output.d[i] > output.d[ans]) {
				ans = i;
			}
			if (target == i) dErr.d[i] -= 1;
		}
		//derive at hidden
		dvec<HIDDEN_SIZE> dHidden;
		MVLeftMultiply(model->Wfc, dErr, dHidden);
		//derive at output gate
		dvec<HIDDEN_SIZE> dGo(dHidden);
		dGo *= cellTanh; dGo *= Go.getDerivSigmoid();
		//derive at cell
		dvec<HIDDEN_SIZE> dCell(dHidden);
		dCell *= Go; dCell *= cellTanh.getDerivTanh();
		//derive at input/forget/hidden gate
		dvec<HIDDEN_SIZE> dGi(dCell); dGi *= Gu; dGi *= Gi.getDerivSigmoid();
		dvec<HIDDEN_SIZE> dGu(dCell); dGu *= Gi; dGu *= Gu.getDerivTanh();
		dvec<HIDDEN_SIZE> dGf(dCell); dGf *= lastState->cell; dGf *= Gf.getDerivSigmoid();

		//derive at input embedding
		d.input.initToZero();
		dvec<EMBEDDING_SIZE> tmp;
		MVLeftMultiply(model->Wi, dGi, tmp); d.input += tmp;
		MVLeftMultiply(model->Wf, dGf, tmp); d.input += tmp;
		MVLeftMultiply(model->Wo, dGo, tmp); d.input += tmp;
		MVLeftMultiply(model->Wu, dGu, tmp); d.input += tmp;
		//derive for bias
		d.Bf = dGf; d.Bi = dGi;
		d.Bo = dGo; d.Bu = dGu;

		//derive for weights
		VVOuterProduct(dGo, *input, d.Wo);
		VVOuterProduct(dGi, *input, d.Wi);
		VVOuterProduct(dGf, *input, d.Wf);
		VVOuterProduct(dGu, *input, d.Wu);
		VVOuterProduct(dGo, lastState->hidden, d.Uo);
		VVOuterProduct(dGi, lastState->hidden, d.Ui);
		VVOuterProduct(dGf, lastState->hidden, d.Uf);
		VVOuterProduct(dGu, lastState->hidden, d.Uu);
		d.Po = dGo; d.Po *= lastState->cell;
		d.Pi = dGi; d.Pi *= lastState->cell;
		d.Pf = dGf; d.Pf *= lastState->cell;
	}
};

struct Sentence {
	std::vector<string> tokens;
	std::vector<int> target;
};
std::vector<Sentence*> trainData, testData;
std::map<string, int> targetMap;
std::vector<string> tagList;
int tagCount = 0;

LSTMTaggerModel model;

void loadTrainData(LSTMTaggerModel &model) {
	std::ifstream fin(TRAIN_FILE);
	char *buf = new char[50];
	Sentence* ns = new Sentence();
	tagList.push_back("*");
	while (fin.getline(buf, 45)) {
		string tmp(buf);
		if (tmp.length() == 0) {
			trainData.push_back(ns);
			ns = new Sentence();
			continue;
		}
		int tokenEnd;
		for (int i = 0; i < tmp.length(); ++i) {
			if (tmp[i] == ' ') {
				tokenEnd = i;
				break;
			}
		}
		string tagStr = tmp.substr(tokenEnd + 1);
		string tokenStr = tmp.substr(0, tokenEnd);
		ns->tokens.push_back(tokenStr);
		if (targetMap[tagStr] == 0) {
			targetMap[tagStr] = ++tagCount;
			tagList.push_back(tagStr);
		}
		ns->target.push_back(targetMap[tagStr]);
		if (model.lookupTable.find(tokenStr) == model.lookupTable.end()) {
			model.lookupTable[tokenStr].initToRandom(5);
		}
	}
	if (ns->tokens.size() != 0) {
		trainData.push_back(ns);
	}
	fin.close();
	if (tagCount+1 != OUTPUT_SIZE) {
		std::cout << "Tag Count = " << tagCount << " while OutputSize=" << OUTPUT_SIZE << "STOP!!!" <<std::endl;
		while (1);
	}
	std::cout << "Load train data from " << TRAIN_FILE << " , total sentence: " << trainData.size() << std::endl;
}

void loadTestData(LSTMTaggerModel &model) {
	std::ifstream fin(TEST_FILE);
	char *buf = new char[50];
	Sentence* ns = new Sentence();
	while (fin.getline(buf, 45)) {
		string tmp(buf);
		if (tmp.length() == 0) {
			testData.push_back(ns);
			ns = new Sentence();
			continue;
		}
		int tokenEnd;
		for (int i = 0; i < tmp.length(); ++i) {
			if (tmp[i] == ' ') {
				tokenEnd = i;
				break;
			}
		}
		string tagStr = tmp.substr(tokenEnd + 1);
		string tokenStr = tmp.substr(0, tokenEnd);
		ns->tokens.push_back(tokenStr);
		if (targetMap[tagStr] == 0) {
			targetMap[tagStr] = ++tagCount;
			tagList.push_back(tagStr);
		}
		ns->target.push_back(targetMap[tagStr]);
		if (model.lookupTable.find(tokenStr) == model.lookupTable.end()) {
			model.lookupTable[tokenStr].initToRandom(5);
		}
	}
	if (ns->tokens.size() != 0) {
		testData.push_back(ns);
	}
	fin.close();
	if (tagCount + 1 != OUTPUT_SIZE) {
		std::cout << "Tag Count = " << tagCount << " while OutputSize=" << OUTPUT_SIZE << std::endl;
		while (1);

	}
	std::cout << "Load test data from " << TRAIN_FILE << " , total sentence: " << testData.size() << std::endl;
}
void testModel(LSTMTaggerModel &model, const char *outputFileName) {
	std::ofstream fout(outputFileName);
	for (int ele = 0; ele < testData.size(); ++ele) {
		Sentence *s = testData[ele];
		LSTMState ** states = new LSTMState*[s->target.size() + 1];
		states[0] = new LSTMState(&model);
		states[0]->init();
		for (int i = 0; i < s->tokens.size(); ++i) {
			states[i + 1] = new LSTMState(&model);
			states[i + 1]->init();
			states[i + 1]->feedForward(states[i], s->tokens[i]);
			fout << s->tokens[i] << " " << tagList[states[i + 1]->getAnswer()] << std::endl;
		}
		fout << std::endl;

		//clean
		for (int i = 0; i < s->target.size() + 1; ++i) {
			delete states[i];
		}
		delete[] states;
		//report
	}
	fout.close();
	std::cout << "Output to file: " << outputFileName << std::endl;
}
void trainModel(LSTMTaggerModel &model) {
	model.init();
	for (int iter = 0; iter < ITER_COUNT; ++iter) {
		int correct = 0, total = 0;
		for (int ele = 0; ele < trainData.size(); ++ele) {
			Sentence *s = trainData[ele];
			total += s->tokens.size();
			LSTMState ** states = new LSTMState*[s->target.size()+1];
			Delta ** deltas = new Delta*[s->target.size()];
			states[0] = new LSTMState(&model);
			states[0]->init();
			for (int i = 0; i < s->tokens.size(); ++i) {
				states[i + 1] = new LSTMState(&model);
				deltas[i] = new Delta();
				states[i + 1]->init();
				states[i + 1]->feedForward(states[i], s->tokens[i]);
				if (states[i + 1]->getAnswer() == s->target[i]) correct+=1;
				states[i + 1]->backProp(s->target[i], *deltas[i]);
			}
			//update parameters
			for (int i = 0; i < s->tokens.size(); ++i) {
				deltas[i]->input *= LEARN_RATE; model.lookupTable[s->tokens[i]] -= deltas[i]->input;
				deltas[i]->Wf *= LEARN_RATE; model.Wf -= deltas[i]->Wf;
				deltas[i]->Wi *= LEARN_RATE; model.Wi -= deltas[i]->Wi;
				deltas[i]->Wo *= LEARN_RATE; model.Wo -= deltas[i]->Wo;
				deltas[i]->Wu *= LEARN_RATE; model.Wu -= deltas[i]->Wu;
				deltas[i]->Uf *= LEARN_RATE; model.Uf -= deltas[i]->Uf;
				deltas[i]->Ui *= LEARN_RATE; model.Ui -= deltas[i]->Ui;
				deltas[i]->Uo *= LEARN_RATE; model.Uo -= deltas[i]->Uo;
				deltas[i]->Uu *= LEARN_RATE; model.Uu -= deltas[i]->Uu;
				deltas[i]->Bf *= LEARN_RATE; model.Bf -= deltas[i]->Bf;
				deltas[i]->Bi *= LEARN_RATE; model.Bi -= deltas[i]->Bi;
				deltas[i]->Bo *= LEARN_RATE; model.Bo -= deltas[i]->Bo;
				deltas[i]->Bu *= LEARN_RATE; model.Bu -= deltas[i]->Bu;
				deltas[i]->Pf *= LEARN_RATE; model.Pf -= deltas[i]->Pf;
				deltas[i]->Pi *= LEARN_RATE; model.Pi -= deltas[i]->Pi;
				deltas[i]->Po *= LEARN_RATE; model.Po -= deltas[i]->Po;
				deltas[i]->Wfc *= LEARN_RATE; model.Wfc -= deltas[i]->Wfc;
			}
			
			//clean
			for (int i = 0; i < s->target.size() + 1; ++i) {
				delete states[i]; 
			}
			for (int i = 0; i < s->target.size(); ++i) {
				delete deltas[i];
			}
			delete[] states;
			delete[] deltas;
			//report
			std::cout << "Sentence " << ele << ": Precious=" << ((double)correct) / total << std::endl;
			
		}
		model.Bf *= SHRINK_RATE; model.Wf *= SHRINK_RATE; model.Uf *= SHRINK_RATE; model.Pf *= SHRINK_RATE;
		model.Bi *= SHRINK_RATE; model.Wi *= SHRINK_RATE; model.Ui *= SHRINK_RATE; model.Pi *= SHRINK_RATE;
		model.Bo *= SHRINK_RATE; model.Wo *= SHRINK_RATE; model.Uo *= SHRINK_RATE; model.Po *= SHRINK_RATE;
		model.Bu *= SHRINK_RATE; model.Wu *= SHRINK_RATE; model.Uu *= SHRINK_RATE; 
		model.Wfc *= SHRINK_RATE;
		std::cout << "Start testing..." << std::endl;
		char filenameBuf[512];
		sprintf_s(filenameBuf, OUTPUT_FILE_NAME_PATTERN, iter);
		testModel(model, filenameBuf);
	}
}


int main(int argc, char **args) {
	loadTrainData(model);
	loadTestData(model);
	trainModel(model);
	return 0;
}