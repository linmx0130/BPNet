#include "BiGRU.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
const char * TRAIN_FILE = "pos_train.utf8";
const char * DEV_FILE = "pos_test.utf8";
const char * OUTPUT_FILE_ROOT = "pos_output_";
struct TokenPair {
	std::string token, tag;
	TokenPair(std::string token, std::string tag):token(token), tag(tag) {}
};
struct Sentence {
	std::vector<TokenPair> tokens;
	size_t size() const{
		return tokens.size();
	}
};

std::vector<Sentence> trainData;
std::vector<Sentence> devData;
std::set<std::string> tokenSet;
std::set<std::string> tagSet;
std::map<std::string, int> tagIdMap;
std::vector<std::string> tagList;
double tagTrans[TAG_SIZE][TAG_SIZE];
void loadDataToList(const char *filename, std::vector<Sentence> & list) {
	std::ifstream fin(filename);
	std::string buf;
	Sentence tmpData;
	while (std::getline(fin, buf)) {
		if (buf.length() == 0) {
			list.push_back(tmpData);
			tmpData.tokens.clear();
			continue;
		}
		std::istringstream istr(buf);
		std::string token, tag;
		istr >> token >> tag;
		tokenSet.insert(token);
		tagSet.insert(tag);
		tmpData.tokens.push_back({ token, tag });
	}
	if (tmpData.tokens.size() != 0) {
		list.push_back(tmpData);
	}
}
void loadData(const char * trainFile,const char* devFile) {
	loadDataToList(trainFile, trainData);
	loadDataToList(devFile, devData);
}
void initTagList() {
	tagList.push_back(START_TAG);
	tagIdMap[START_TAG] = 0;
	for (auto iter = tagSet.cbegin(); iter != tagSet.cend(); ++iter) {
		tagList.push_back(*iter);
		tagIdMap[*iter] = tagList.size() - 1;
	}
}
NetworkParam network;
GRUStatus<EMBED_SIZE, HIDDEN_SIZE> emptyGRUStatus;
void initLookUpTable() {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
	for (auto iter = tokenSet.cbegin(); iter != tokenSet.cend(); ++iter) {
		dvec<EMBED_SIZE> *tmp = new dvec<EMBED_SIZE>;
		tmp->initToRandom(1);
		network.lookUpTable[*iter] = tmp;
	}
}
void feedForward(Sentence& sentence, SentenceStatus & status) {
	status.clear();
	for (int i = 0; i < sentence.size(); ++i) {
		TokenStatus *ts = new TokenStatus();
		ts->tokenEmbed = network.lookUpTable[sentence.tokens[i].token];
		status.items.push_back(ts);
	}
	// left-to-right gru
	for (int i = 0; i < sentence.size(); ++i) {
		GRUStatus<EMBED_SIZE, HIDDEN_SIZE> & lastGRUStatus = (i == 0) ? emptyGRUStatus : status.items[i-1]->gru1;
		network.gru1.forward(*status.items[i]->tokenEmbed, lastGRUStatus, status.items[i]->gru1);
	}
	// right-to-left gru
	for (int i = sentence.size()-1; i >=0; --i) {
		GRUStatus<EMBED_SIZE, HIDDEN_SIZE> & lastGRUStatus = (i == sentence.size()-1) ? emptyGRUStatus : status.items[i + 1]->gru2;
		network.gru2.forward(*status.items[i]->tokenEmbed, lastGRUStatus, status.items[i]->gru2);
	}
	// later layers
	for (int i = 0; i < sentence.size(); ++i) {
		VConcat(status.items[i]->gru1.h, status.items[i]->gru2.h, status.items[i]->concat);
		MVRightMultiply(network.W1, status.items[i]->concat, status.items[i]->fc1);
		status.items[i]->fc1 += network.B1;
		status.items[i]->fc1.performSoftmax();
	}
}
void derivBackward(SentenceStatus & status, const std::vector<dvec<TAG_SIZE>> dLoss, SentenceGradient &g) {
	//prepare gradient space
	g.clear();
	g.status = &status;
	dvec<HIDDEN_SIZE> *dGru1, *dGru2;
	dGru1 = new dvec<HIDDEN_SIZE>[status.size()];
	dGru2 = new dvec<HIDDEN_SIZE>[status.size()];
	for (int i = 0; i < status.size(); ++i) {
		TokenGradient *tmp = new TokenGradient();
		g.items.push_back(tmp);
	}
	//deriv Concat layer, FC1 and FC2
	for (int i = 0; i < status.size(); ++i) {
		dvec<FC1_SIZE> dFc1;
		dvec<HIDDEN_SIZE*2> dConcat;
		dFc1 = dLoss[i];
		network.B1 = dFc1;
		MVLeftMultiply(network.W1, dFc1, dConcat);
		VCopySegment(dConcat, 0, dGru1[i]);
		VCopySegment(dConcat, HIDDEN_SIZE, dGru2[i]);
		VVOuterProduct(dFc1, status.items[i]->concat, g.items[i]->W1);
	}
	//deriv GRU
	for (int i = 0; i < status.size(); ++i) {
		network.gru2.backward(dGru2[i], status.items[i]->gru2, g.items[i]->gru2);
		if (i != status.size() - 1) {
			dGru2[i + 1] += g.items[i]->gru2.lastH;
		}
	}
	for (int i = status.size() - 1; i >= 0; --i) {
		network.gru1.backward(dGru1[i], status.items[i]->gru1, g.items[i]->gru1);
		if (i != 0) {
			dGru1[i -1] += g.items[i]->gru1.lastH;
		}
	}
	delete[] dGru1;
	delete[] dGru2;
}
int softmaxDecoding(const Sentence & sentence, SentenceStatus & status, std::vector<dvec<TAG_SIZE>> & dLoss) {
	int count = status.size();
	dLoss.clear();
	int correct = 0;
	for (int i = 0; i < count; ++i) {
		int target = tagIdMap[sentence.tokens[i].tag];
		int answer = 0;
		for (int t = 0; t < TAG_SIZE; ++t) {
			if (status.items[i]->fc1.d[t]>status.items[i]->fc1.d[answer]) {
				answer = t;
			}
		}
		if (answer == target) correct++;
		dvec<TAG_SIZE> dLossOnToken = status.items[i]->fc1;
		dLossOnToken.d[target]-=1;
		dLoss.push_back(dLossOnToken);
	}
	return correct;
}
void softmaxDecoding(SentenceStatus & status) {
	int count = status.size();
	status.predictLabel = new int[count];
	int correct = 0;
	for (int i = 0; i < count; ++i) {
		int answer = 0;
		for (int t = 0; t < TAG_SIZE; ++t) {
			if (status.items[i]->fc1.d[t]>status.items[i]->fc1.d[answer]) {
				answer = t;
			}
		}
		status.predictLabel[i] = answer;
	}
}
/* return the correct count of the data*/
int maxMarginDecoding(const Sentence & sentence, SentenceStatus & status, std::vector<dvec<TAG_SIZE>> & dLoss) {
	int count = sentence.size();
	dLoss.clear();
	int correct = 0;
	double ** scoreF = new double*[count];
	int **lastTag = new int*[count];
	int *predictLabel = new int[count];
	for (int i = 0; i < count; ++i) {
		scoreF[i] = new double[TAG_SIZE];
		lastTag[i] = new int[TAG_SIZE];
		for (int j = 0; j < TAG_SIZE; ++j) { 
			scoreF[i][j] = -10000; 
			lastTag[i][j] = 0;
		}
	}
	{
		int target = tagIdMap[sentence.tokens[0].tag];
		for (int j = 0; j < TAG_SIZE; ++j) {
			double tmpScore = tagTrans[0][j] + status.items[0]->fc1.d[j];
			if (target != j) tmpScore += MARGIN;
			scoreF[0][j] = tmpScore;
		}
	}
	for (int i = 1; i < count; ++i) {
		int target = tagIdMap[sentence.tokens[i].tag];
		for (int j = 0; j < TAG_SIZE; ++j) {
			for (int k = 0; k < TAG_SIZE; ++k) {
				double tmpScore = status.items[i-1]->fc1.d[k] + tagTrans[k][j] + status.items[i]->fc1.d[j];
				if (target != j) tmpScore += MARGIN;
				if (tmpScore > scoreF[i][j]) {
					scoreF[i][j] = tmpScore;
					lastTag[i][j] = k;
				}
			}
		}
	}
	//decoding 
	predictLabel[count -1] = 0;
	for (int j = 0; j < TAG_SIZE; ++j) {
		if (status.items[count - 1]->fc1.d[j] > status.items[count - 1]->fc1.d[predictLabel[count - 1]]) {
			predictLabel[count - 1] = j;
		}
	}
	for (int i = count - 2; i >= 0; --i) {
		predictLabel[i] = lastTag[i + 1][predictLabel[i + 1]];
	}
	for (int i = 0; i < count;++i){
		int answer = predictLabel[i];
		int target = tagIdMap[sentence.tokens[i].tag];
		if (answer == target) correct++;
		//::cout << answer << "/" << target << " ";
		dvec<TAG_SIZE> dLossOnToken; dLossOnToken.initToZero();
		dLossOnToken.d[answer]++;
		dLossOnToken.d[target]--;
		dLoss.push_back(dLossOnToken);
	}
	for (int i = 0; i < count; ++i) {
		delete[] scoreF[i];
		delete[] lastTag[i];
	}
	delete[] scoreF;
	delete[] lastTag;
	status.predictLabel = predictLabel;
	//std::cout <<"Length:" << sentence.size()<< std::endl;
	return correct;
} 

void normalDecoding(SentenceStatus & status) {
	int count = status.size();
	double ** scoreF = new double*[count];
	int **lastTag = new int*[count];
	int *predictLabel = new int[count];
	for (int i = 0; i < count; ++i) {
		scoreF[i] = new double[TAG_SIZE];
		lastTag[i] = new int[TAG_SIZE];
		for (int j = 0; j < TAG_SIZE; ++j) {
			scoreF[i][j] = -10000;
			lastTag[i][j] = 0;
		}
	}
	{
		for (int j = 0; j < TAG_SIZE; ++j) {
			double tmpScore = tagTrans[0][j] + status.items[0]->fc1.d[j];
			scoreF[0][j] = tmpScore;
		}
	}
	for (int i = 1; i < count; ++i) {
		for (int j = 0; j < TAG_SIZE; ++j) {
			for (int k = 0; k < TAG_SIZE; ++k) {
				double tmpScore = status.items[i - 1]->fc1.d[k] + tagTrans[k][j] + status.items[i]->fc1.d[j];
				if (tmpScore > scoreF[i][j]) {
					scoreF[i][j] = tmpScore;
					lastTag[i][j] = k;
				}
			}
		}
	}
	//decoding 
	predictLabel[count - 1] = 0;
	for (int j = 0; j < TAG_SIZE; ++j) {
		if (status.items[count - 1]->fc1.d[j] > status.items[count - 1]->fc1.d[predictLabel[count - 1]]) {
			predictLabel[count - 1] = j;
		}
	}
	for (int i = count - 2; i >= 0; --i) {
		predictLabel[i] = lastTag[i + 1][predictLabel[i + 1]];
	}
	for (int i = 0; i < count; ++i) {
		delete[] scoreF[i];
		delete[] lastTag[i];
	}
	delete[] scoreF;
	delete[] lastTag;
	status.predictLabel = predictLabel;
}
template <int INPUT_SIZE, int HIDDEN_SIZE>
void updateGRU(GRUParam<INPUT_SIZE, HIDDEN_SIZE> & gru, const GRUGradient<INPUT_SIZE, HIDDEN_SIZE> &gradient, double learningRate) {
	gru.U -= gradient.U * learningRate;
	gru.W -= gradient.W * learningRate;
	gru.Wr -= gradient.Wr *learningRate;
	gru.Ur -= gradient.Ur * learningRate;
	gru.Uz -= gradient.Uz * learningRate;
	gru.Wz -= gradient.Wz * learningRate;
	gru.Bz -= gradient.Bz*learningRate;
	gru.Br -= gradient.Br *learningRate;
	gru.Bhn -= gradient.Bhn*learningRate;
}
void updateLookUpTable(const std::string & key, const dvec<EMBED_SIZE> dValue, double learningRate) {
	auto &v = *network.lookUpTable[key];
	v -= dValue *learningRate;
}
void updateNetwork(const TokenGradient &gradient, double learningRate) {
	network.W1 -= gradient.W1 * learningRate;
	network.B1 -= gradient.B1*learningRate;
	updateGRU(network.gru1, gradient.gru1, learningRate*2);
	updateGRU(network.gru2, gradient.gru2, learningRate*2);
}
void updateTrans(const SentenceStatus& status, const Sentence& sentence, double learningRate) {
	int count = status.size();
	int *l = status.predictLabel;
	for (int i = 1; i < count; ++i) {
		tagTrans[l[i - 1]][l[i]] -= learningRate;
		tagTrans[tagIdMap[sentence.tokens[i - 1].tag]][tagIdMap[sentence.tokens[i].tag]] += learningRate;
	}
}
void testNetwork(const char* outputFile) {
	SentenceStatus status;
	std::ofstream fout(outputFile);
	for (int item = 0; item < devData.size();++item) {
		int correctCount = 0;
		int totalCount = 0;
		status.clear();
		feedForward(devData[item], status);
		softmaxDecoding(status);
		for (int i = 0; i < status.size(); ++i) {
			fout << devData[item].tokens[i].token << " " << tagList[status.predictLabel[i]] << std::endl;
		}
		fout << std::endl;
	}
	fout.close();
}
void trainNetwork() {
	SentenceStatus status[BATCH_SIZE];
	SentenceGradient gradient[BATCH_SIZE];
	std::vector<dvec<TAG_SIZE>> dLoss;
	for (int iter = 0; iter < ITER_COUNT; ++iter) {
		std::cout << "ITER " << iter << std::endl;
		for (int item = 0; item < trainData.size();) {
			int current;
			int correctCount = 0;
			int totalCount = 0;
			for (current = 0; current < BATCH_SIZE && item < trainData.size(); ++current, ++item) {
				feedForward(trainData[item], status[current]);
				correctCount += softmaxDecoding(trainData[item], status[current], dLoss);
				totalCount += trainData[item].size();
				derivBackward(status[current], dLoss, gradient[current]);
			}
			int batch = current;
			item -= batch;
			for (current = 0; current < batch; ++current, ++item) {
				int length = status[current].size();
				double learningRate = LEARN_RATE; // length;
				//updateTrans(status[current], trainData[item], learningRate);
				for (int i = 0; i <length; ++i) {
					updateNetwork(*gradient[current].items[i], learningRate);
					updateLookUpTable(trainData[item].tokens[i].token, gradient[current].items[i]->gru1.inputVec ,learningRate*4);
					updateLookUpTable(trainData[item].tokens[i].token, gradient[current].items[i]->gru2.inputVec, learningRate*4);
				}
			}
			std::cout << "Items " << item - batch << " ~ " << item - 1
				<< " Tag Correctness: " << ((double)correctCount) / totalCount << std::endl;
		}
		std::ostringstream outputFileBuf;
		outputFileBuf << OUTPUT_FILE_ROOT << iter << ".utf8";
		outputFileBuf.str().c_str();
		testNetwork(outputFileBuf.str().c_str());
	}
}
#include <ctime>
int main(int argc, char** argv) {
	srand(time(0));
	std::cout << "LOADING DATA..." << std::endl;
	loadData(TRAIN_FILE, DEV_FILE);
	std::cout << "INIT PARAMETERS" << std::endl;
	initLookUpTable();
	initTagList();
	network.init();
	std::cout << "START TRAINING" << std::endl;
	trainNetwork();
	return 0;
}