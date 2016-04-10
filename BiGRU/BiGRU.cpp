#include "BiGRU.h"
const char* START_TAG = "<START>";
void NetworkParam::init() {
	gru1.init();
	gru2.init();
	W1.initToRandom(sqrt(W1.getColumnSize()));
	B1.initToRandom(sqrt(W1.getColumnSize()));
}