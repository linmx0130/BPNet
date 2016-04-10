#ifndef _GRU_H
#define _GRU_H

#include "cornernn_math.h"
using namespace cornernn;
template <int ROW, int COLUMN>
using dmat = cornernn::Matrix<double, ROW, COLUMN>;

template <int LEN>
using dvec = cornernn::Vector<double, LEN>;

template <int INPUT_SIZE, int HIDDEN_SIZE>
struct GRUStatus {
	dvec<HIDDEN_SIZE> r, z, hn, h;
	dvec<INPUT_SIZE> *input;
	GRUStatus *lastStatus;
	void init() {
		r.initToZero();
		z.initToZero();
		hn.initToZero();
		h.initToZero();
		lastStatus = nullptr;
		input = nullptr;
	}
};
template <int INPUT_SIZE, int HIDDEN_SIZE>
struct GRUGradient {
	dmat<HIDDEN_SIZE, HIDDEN_SIZE>  U, Ur, Uz;
	dmat<HIDDEN_SIZE, INPUT_SIZE> W, Wr, Wz;
	dvec<HIDDEN_SIZE> lastH;
	dvec<INPUT_SIZE> inputVec;
	dvec<HIDDEN_SIZE> Bz, Br, Bhn;
	GRUStatus<INPUT_SIZE, HIDDEN_SIZE> * status;
	void init() {
		U.initToZero(); Ur.initToZero(); Uz.initToZero();
		W.initToZero(); Wr.initToZero(); Wz.initToZero();
		lastH.initToZero();
		inputVec.initToZero();
		Bz.initToZero(); Br.initToZero(); Bhn.initToZero();
		status = nullptr;
	}
};

template <int INPUT_SIZE, int HIDDEN_SIZE>
struct GRUParam {
	dmat<HIDDEN_SIZE, HIDDEN_SIZE>  U, Ur, Uz;
	dmat<HIDDEN_SIZE, INPUT_SIZE> W, Wr, Wz;
	dvec<HIDDEN_SIZE> Bz, Br, Bhn;
	void forward(dvec<INPUT_SIZE>& inputVec, 
		GRUStatus<INPUT_SIZE,HIDDEN_SIZE> & lastStatus, 
		GRUStatus<INPUT_SIZE, HIDDEN_SIZE> & status) {
		//init status
		status.init();
		status.lastStatus = &lastStatus;
		status.input = &inputVec;
		dvec<HIDDEN_SIZE> tmp;

		//reset gate
		MVRightMultiply(Wr, inputVec, tmp);
		MVRightMultiply(Ur, lastStatus.h, status.r);
		status.r += tmp;
		status.r += Br;
		status.r.performSigmoid();

		//update gate
		MVRightMultiply(Wz, inputVec, tmp);
		MVRightMultiply(Uz, lastStatus.h, status.z);
		status.z += tmp;
		status.z += Bz;
		status.z.performSigmoid();

		//hidden step
		tmp = lastStatus.h;
		tmp *= status.r; 
		MVRightMultiply(U, tmp, status.hn);
		MVRightMultiply(W, inputVec, tmp);
		status.hn += tmp;
		status.hn += Bhn;
		status.hn.performTanh();

		//hidden
		status.h = status.hn;
		status.h -= lastStatus.h;
		status.h *= status.z;
		status.h += lastStatus.h;
	}
	void backward(dvec<HIDDEN_SIZE> &gHidden, 
		GRUStatus<INPUT_SIZE, HIDDEN_SIZE> & status, 
		GRUGradient<INPUT_SIZE, HIDDEN_SIZE> & g) {
		g.init();
		g.status = &status;
		// temp variables
		dvec<HIDDEN_SIZE> tmp;
		dvec<HIDDEN_SIZE> gHn;
		dvec<HIDDEN_SIZE> gZ;
		dvec<HIDDEN_SIZE> gR;

		// deriv gHn to Wx+U(r*h)
		gHn = gHidden * status.z;
		gHn *= status.hn.getDerivTanh();
		g.Bhn = gHn;
		//deriv gZ
		gZ = status.hn;
		gZ -= status.lastStatus->h;
		gZ *= gHidden;
		gZ *= status.z.getDerivSigmoid();
		g.Bz = gZ;
		//deriv gR
		MVLeftMultiply(U, gHn, gR);
		gR *= status.lastStatus->h;
		gR *= status.r.getDerivSigmoid();
		g.Br = gR;
		//deriv last hidden
		MVLeftMultiply(Uz, gZ, g.lastH);
		MVLeftMultiply(Ur, gR, tmp); g.lastH += tmp;
		MVLeftMultiply(U, gHn, tmp);
		tmp *= status.r;
		g.lastH += tmp;
		g.lastH += gHidden;
		g.lastH -= gHidden * status.z;
		//deriv input 
		MVLeftMultiply(Wz, gZ, g.inputVec);
		MVLeftMultiply(Wr, gR, tmp); g.inputVec += tmp;
		MVLeftMultiply(W, gHn, tmp); g.inputVec += tmp;
		//deriv matrices
		VVOuterProduct(gZ, *status.input, g.Wz);
		VVOuterProduct(gZ, status.lastStatus->h, g.Uz);
		VVOuterProduct(gR, *status.input, g.Wr);
		VVOuterProduct(gR, status.lastStatus->h, g.Ur);
		tmp = status.r;
		tmp *= status.lastStatus->h;
		VVOuterProduct(gHn, tmp, g.U);
		VVOuterProduct(gHn, *status.input, g.W);
	}
	void init() {
		W.initToRandom(sqrt(INPUT_SIZE));
		Wr.initToRandom(sqrt(INPUT_SIZE));
		Wz.initToRandom(sqrt(INPUT_SIZE));
		U.initToRandom(sqrt(HIDDEN_SIZE));
		Ur.initToRandom(sqrt(HIDDEN_SIZE));
		Uz.initToRandom(sqrt(HIDDEN_SIZE));
		Bz.initToRandom(sqrt(HIDDEN_SIZE));
		Br.initToRandom(sqrt(HIDDEN_SIZE));
		Bhn.initToRandom(sqrt(HIDDEN_SIZE));
	}
};

#endif 
