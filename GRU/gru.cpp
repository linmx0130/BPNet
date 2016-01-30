#include "cornernn_math.h"
#include <iostream>
#include <fstream>

using namespace cornernn;
const int INPUT_SIZE = 3;
const int HIDDEN_SIZE = 10;
const int OUTPUT_SIZE = 1;
Matrix<double, HIDDEN_SIZE, INPUT_SIZE> Wz, Wr,Wh;
Matrix<double, HIDDEN_SIZE, HIDDEN_SIZE> Uz, Ur,Uh;
Matrix<double, OUTPUT_SIZE, HIDDEN_SIZE> Wo;
Vector<double, INPUT_SIZE> inputVec;
Vector<double, HIDDEN_SIZE> hVec, zVec, rVec, h2Vec;
Vector<double, OUTPUT_SIZE> oVec;
const double LEARNING_RATE = 0.1;
const int ITER_COUNT = 150;

void init() {
	Wz.initToRandom(sqrt(Wz.getColumnSize()));
	Wr.initToRandom(sqrt(Wr.getColumnSize()));
	Uz.initToRandom(sqrt(Uz.getColumnSize()));
	Ur.initToRandom(sqrt(Ur.getColumnSize()));
	Wh.initToRandom(sqrt(Wh.getColumnSize()));
	Uh.initToRandom(sqrt(Uh.getColumnSize()));
	Wo.initToRandom(sqrt(Wo.getColumnSize()));
	hVec.initToZero();
}
void feedForward() {
	Vector<double, HIDDEN_SIZE> tmp1;
	MVRightMultiply(Wz, inputVec, zVec);
	MVRightMultiply(Uz, hVec, tmp1);
	zVec += tmp1;
	zVec.performSigmoid();

	MVRightMultiply(Wr, inputVec, rVec);
	MVRightMultiply(Ur, hVec, tmp1);
	rVec += tmp1;
	rVec.performSigmoid();
	
	tmp1 = hVec;
	tmp1 *= rVec;
	MVRightMultiply(Uh, tmp1, h2Vec);
	MVRightMultiply(Wh, inputVec, tmp1);
	h2Vec += tmp1;
	h2Vec.performTanh();
	
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		hVec.d[i] = (1 - zVec.d[i])*hVec.d[i] + zVec.d[i] * h2Vec.d[i];
	}

	MVRightMultiply(Wo, hVec, oVec);
}
double trainBack(const Vector<double,INPUT_SIZE> & input, const Vector<double,OUTPUT_SIZE> & target) {
	Vector<double, HIDDEN_SIZE> oldHVec(hVec);
	inputVec = input;
	Vector<double, HIDDEN_SIZE> tmp1, rHProduct;
	MVRightMultiply(Wz, inputVec, zVec);
	MVRightMultiply(Uz, hVec, tmp1);
	zVec += tmp1;
	zVec.performSigmoid();

	MVRightMultiply(Wr, inputVec, rVec);
	MVRightMultiply(Ur, hVec, tmp1);
	rVec += tmp1;
	rVec.performSigmoid();

	tmp1 = hVec;
	tmp1 *= rVec;
	rHProduct = tmp1;
	MVRightMultiply(Uh, tmp1, h2Vec);
	MVRightMultiply(Wh, inputVec, tmp1);
	h2Vec += tmp1;
	h2Vec.performTanh();

	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		hVec.d[i] = (1 - zVec.d[i])*hVec.d[i] + zVec.d[i] * h2Vec.d[i];
	}
	MVRightMultiply(Wo, hVec, oVec);
	
	//get error
	double errorValue = 0;
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		errorValue += (target.d[i] - oVec.d[i])* (target.d[i] - oVec.d[i]);
	}
	//get deriv
	Vector<double, OUTPUT_SIZE> dError(oVec);
	dError -= target;
	Vector<double, HIDDEN_SIZE> dHVec;
	MVLeftMultiply(Wo, dError, dHVec);
	
	Vector<double, HIDDEN_SIZE> dZVec;
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		dZVec.d[i] = dHVec.d[i] * (-oldHVec.d[i] + h2Vec.d[i]);
	}
	dZVec *= zVec.getDerivSigmoid();

	Vector<double, HIDDEN_SIZE> dH2Vec;
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		dH2Vec.d[i] = dHVec.d[i] * zVec.d[i];
	}
	dH2Vec *= h2Vec.getDerivTanh();

	Vector<double, HIDDEN_SIZE> dRHVec;
	MVLeftMultiply(Uh, dH2Vec, dRHVec);

	Vector<double, HIDDEN_SIZE> dRVec;
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		dRVec.d[i] = dRHVec.d[i] * oldHVec.d[i];
	}
	dRVec *= rVec.getDerivSigmoid();
	//update 
	for (int i = 0; i < OUTPUT_SIZE; ++i) {
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			Wo.d[i][j] -= LEARNING_RATE*dError.d[i] * hVec.d[j];
		}
	}
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		for (int j = 0; j < INPUT_SIZE; ++j) {
			Wz.d[i][j] -= LEARNING_RATE * dZVec.d[i] * inputVec.d[j];
		}
	}
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			Uz.d[i][j] -= LEARNING_RATE * dZVec.d[i] * oldHVec.d[j];
		}
	}
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		for (int j = 0; j < INPUT_SIZE; ++j) {
			Wr.d[i][j] -= LEARNING_RATE * dRVec.d[i] * inputVec.d[j];
		}
	}
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			Ur.d[i][j] -= LEARNING_RATE * dRVec.d[i] * oldHVec.d[j];
		}
	}
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		for (int j = 0; j < INPUT_SIZE; ++j) {
			Wh.d[i][j] -= LEARNING_RATE * dH2Vec.d[i] * inputVec.d[j];
		}
	}
	for (int i = 0; i < HIDDEN_SIZE; ++i) {
		for (int j = 0; j < HIDDEN_SIZE; ++j) {
			Uh.d[i][j] -= LEARNING_RATE * dH2Vec.d[i] * rHProduct.d[j];
		}
	}

	return errorValue;
}
void genData();
void networkRun() {
	init();
	int N, M;
	for (int iter = 0; iter < ITER_COUNT; ++iter) {
		std::cout << "Iter " << iter << ": " << std::endl;
		std::ifstream fin("data.in");
		fin >> N >> M;
		double errorSum = 0;
		for (int i = 0; i < N; ++i) {
			hVec.initToZero();
			double error = 0;
			for (int j = 0; j < M; ++j) {
				Vector<double, INPUT_SIZE> inVec;
				fin >> inVec.d[0] >> inVec.d[1];
				inVec.d[2] = 1;
				Vector<double, OUTPUT_SIZE> outVec;
				fin >> outVec.d[0];
				error+=trainBack(inVec, outVec);
				//std::cout << "    " << inVec.d[0] << ", " << inVec.d[1] << ", " << outVec.d[0] <<",  "<< oVec.d[0] << std::endl;
			}
			errorSum += error;
		}
		std::cout << " error=" << errorSum << std::endl;
		fin.close();
	}
}
int main() {
	//genData();
	networkRun();
	//init();
	hVec.initToZero();
	while (1) {
		std::cout << " Input a(0,1) and b(0,1), -1 for reset... " << std::endl;
		int a, b;
		std::cin >> a;
		if (a == -1) {
			hVec.initToZero();
			continue;
		}
		std::cin >> b;
		inputVec.d[0] = a;
		inputVec.d[1] = b;
		inputVec.d[2] = 1;
		feedForward();
		std::cout << " Output = " << oVec.d[0] << std::endl;
	}
	while (1);
	return 0;
} 