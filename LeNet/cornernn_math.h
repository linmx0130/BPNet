#ifndef CORNERNN_MATH_H
#define CORNERNN_MATH_H
#include "stdc.h"

namespace cornernn {
	template <typename FLOAT_TYPE> FLOAT_TYPE sigmoid(FLOAT_TYPE v) {
		return 1 / (1 + exp((double) (-v)));
	};
	template <> double sigmoid(double v);
	template <> float sigmoid(float v);
	template <typename FLOAT_TYPE> FLOAT_TYPE derivSigmoid(FLOAT_TYPE v) {
		return v * (1.0 - v);
	};
	template<typename FLOAT_TYPE> FLOAT_TYPE ReLU(FLOAT_TYPE x)
	{ 
		return x > 0 ? x : 0; 
	}
	template <typename FLOAT_TYPE> FLOAT_TYPE tanh(FLOAT_TYPE v) {
		return std::tanh((double)v);
	};
	template <>	double tanh(double v);
	template <>	float tanh(float v);

	template <typename FLOAT_TYPE> FLOAT_TYPE derivTanh(FLOAT_TYPE v) {
		return 1.0 - v*v;
	};

	template<typename FLOAT_TYPE, int LEN>
	struct Vector {
		FLOAT_TYPE d[LEN];
		Vector() {}
		Vector(const Vector<FLOAT_TYPE, LEN>& v) {
			for (int i = 0; i < LEN; ++i) {
				d[i] = v.d[i];
			}
		}
		constexpr int getLength() const { return LEN; }
		void initToZero() {
			std::memset(d, 0, sizeof(d));
		}
		void initToRandom(FLOAT_TYPE divisor) {
			for (int i = 0; i < LEN; ++i) {
				d[i][j] = Math.sqrt((rand() % 10000 / 10000)) - 0.5;
				d[i][j] *= 2;
				d[i][j] / divisor;
			}
		}
		Vector<FLOAT_TYPE,LEN> & operator+=(const Vector<FLOAT_TYPE, LEN> & v) {
			for (int i = 0; i < LEN; ++i) {
				d[i] += v.d[i];
			}
			return *this;
		}
		Vector<FLOAT_TYPE, LEN> & operator-=(const Vector<FLOAT_TYPE, LEN> & v) {
			for (int i = 0; i < LEN; ++i) {
				d[i] -= v.d[i];
			}
			return *this;
		}
		Vector<FLOAT_TYPE,LEN> & operator=(const Vector<FLOAT_TYPE, LEN> & v) {
			for (int i = 0; i < LEN; ++i) {
				d[i] = v.d[i];
			}
			return *this;
		}
		Vector<FLOAT_TYPE, LEN> &operator*=(const Vector<FLOAT_TYPE, LEN> &v) {
			for (int i = 0; i < LEN; ++i) {
				d[i] *= v.d[i];
			}
			return *this;
		}
		template<typename FUNCTYPE>
		void perform(const FUNCTYPE &f) {
			for (int i = 0; i < LEN; ++i) {
				d[i] = f(d[i]);
			}
		}
		void performSigmoid() {
			for (int i = 0; i < LEN; ++i) {
				d[i] = sigmoid(d[i]);
			}
		}

		void performTanh() {
			for (int i = 0; i < LEN; ++i) {
				d[i] = tanh(d[i]);
			}
		}

		void performDerivSigmoid() {
			for (int i = 0; i < LEN; ++i) {
				d[i] = derivSigmoid(d[i]);
			}
		}

		void performDerivTanh() {
			for (int i = 0; i < LEN; ++i) {
				d[i] = derivTanh(d[i]);
			}
		}
		Vector<FLOAT_TYPE, LEN> getDerivSigmoid() {
			Vector<FLOAT_TYPE, LEN> ret(*this);
			ret.performDerivSigmoid();
			return ret;
		}
		Vector<FLOAT_TYPE, LEN> getDerivTanh() {
			Vector<FLOAT_TYPE, LEN> ret(*this);
			ret.performDerivTanh();
			return ret;
		}
		void performSoftmax() {
			double sum = 0;
			double maxValue = d[0];
			for (int i = 0; i < LEN; ++i) {
				maxValue = (maxValue > d[i]) ? maxValue : d[i];
			}
			for (int i = 0; i < LEN; ++i) {
				d[i] = exp(d[i]-maxValue);
				sum += d[i];
			}
			for (int i = 0; i < LEN; ++i) {
				d[i] /= sum;
			}
		}
	};
	
	template<typename FLOAT_TYPE, int ROW, int COLUMN>
	struct Matrix {
		FLOAT_TYPE d[ROW][COLUMN];
		constexpr int getRowSize() const { return ROW; }
		constexpr int getColumnSize() const { return COLUMN; }
		void initToZero() {
			memset(d, 0, sizeof(d));
		}
		void initToRandom(FLOAT_TYPE divisor) {
			for (int i = 0; i < ROW; ++i) {
				for (int j = 0; j < COLUMN; ++j) {
					d[i][j] = ((FLOAT_TYPE)(rand() % 10000)-5000) / 5000 /divisor;
				}
			}
		}

		Matrix() {}
		Matrix(const Matrix<FLOAT_TYPE,ROW,COLUMN>& m) {
			for (int i = 0; i < ROW; ++i) {
				for (int j = 0; j < COLUMN; ++j) {
					d[i][j] = m.d[i][j];
				}
			}
		}
		Matrix <FLOAT_TYPE, ROW, COLUMN> & operator += (const Matrix<FLOAT_TYPE, ROW, COLUMN> & m) {
			for (int i = 0; i < ROW; ++i) {
				for (int j = 0; j < COLUMN; ++j) {
					d[i][j] += m.d[i][j];
				}
			}
			return *this;
		}
		Matrix <FLOAT_TYPE, ROW, COLUMN> & operator -= (const Matrix<FLOAT_TYPE, ROW, COLUMN> & m) {
			for (int i = 0; i < ROW; ++i) {
				for (int j = 0; j < COLUMN; ++j) {
					d[i][j] -= m.d[i][j];
				}
			}
			return *this;
		}
		
		Matrix<FLOAT_TYPE, ROW, COLUMN> &operator*= (FLOAT_TYPE v) {
			for (int i = 0; i < ROW; ++i) {
				for (int j = 0; j < COLUMN; ++j) {
					d[i][j] *= v;
				}
			}
			return *this;
		}
		template<typename FUNCTYPE>
		void perform(FUNCTYPE f) {
			for (int i = 0; i < ROW; ++i) {
				for (int j = 0; j < COLUMN; ++j) {
					d[i][j] = f(d[i][j]);
				}
			}
		}
	};

	template<typename FLOAT_TYPE, int ROW,int COLUMN>
	void MVRightMultiply(const Matrix<FLOAT_TYPE, ROW, COLUMN> & m, const Vector<FLOAT_TYPE,COLUMN> & v1, Vector <FLOAT_TYPE, ROW> & out) {
		for (int i = 0; i < ROW; ++i) {
			out.d[i] = 0;
			for (int j = 0; j < COLUMN; ++j) {
				out.d[i] += m.d[i][j] * v1.d[j];
			}
		}
	}

	template<typename FLOAT_TYPE, int ROW, int COLUMN>
	void MVLeftMultiply(const Matrix<FLOAT_TYPE, ROW, COLUMN> & m, const Vector<FLOAT_TYPE, ROW> & v1, Vector <FLOAT_TYPE, COLUMN> & out) {
		for (int i = 0; i < COLUMN; ++i) {
			out.d[i] = 0;
			for (int j = 0; j < ROW; ++j) {
				out.d[i] += m.d[j][i] * v1.d[j];
			}
		}
	}

	template <typename FLOAT_TYPE, int N, int F, int S>
	void MMConv(const Matrix<FLOAT_TYPE, N, N> & input,
		const Matrix<FLOAT_TYPE, F, F> & filter,
		Matrix<FLOAT_TYPE, (N - F +1 ) / S, (N - F +1 ) / S> &out)
	{
		int outLen = out.getColumnSize();
		out.initToZero();
		for (int x = 0; x < outLen; x++) {
			for (int y = 0; y < outLen; y++) {
				for (int i = 0; i < F; ++i) {
					for (int j = 0; j < F; ++j) {
						out.d[x][y] += input.d[x*S + i][y*S + j] * filter.d[i][j];
					}
				}
			}
		}
	}
	
	template <typename FLOAT_TYPE, int N, int F, int S>
	void MMInvConv(const Matrix<FLOAT_TYPE, (N - F+1 ) / S , (N - F +1 ) / S> &input,
		const Matrix<FLOAT_TYPE, F, F> & filter,
		Matrix<FLOAT_TYPE, N, N> & out) {
		int inLen = input.getColumnSize();
		out.initToZero();
		for (int x = 0; x < inLen; x++) {
			for (int y = 0; y < inLen; y++) {
				for (int i = 0; i < F; ++i) {
					for (int j = 0; j < F; ++j) {
						out.d[x*S + i][y*S + j] += input.d[x][y] * filter.d[i][j];
					}
				}
			}
		}
	}
	template <typename FLOAT_TYPE, int N, int S>
	void MeanPooling(const Matrix<FLOAT_TYPE, N, N> & input, Matrix<FLOAT_TYPE, N / S, N / S> & output) {
		output.initToZero();
		for (int x = 0; x < output.getColumnSize(); ++x) {
			for (int y = 0; y < output.getColumnSize(); ++y) {
				for (int i = 0; i < S; ++i) {
					for (int j = 0; j < S; ++j) {
						output.d[x][y] += input.d[x*S + i][y*S + j];
					}
				}
				output.d[x][y] /= S*S;
			}
		}
	}
	template <typename FLOAT_TYPE, int N, int S>
	void InvMeanPooling(const Matrix<FLOAT_TYPE, N / S, N / S> & input, Matrix<FLOAT_TYPE, N, N> & output ) {
		output.initToZero();
		double dividor = S*S;
		for (int x = 0; x < input.getColumnSize(); ++x) {
			for (int y = 0; y < input.getColumnSize(); ++y) {
				for (int i = 0; i < S; ++i) {
					for (int j = 0; j < S; ++j) {
						output.d[x*S+i][y*S+j] += input.d[x][y]/dividor;
					}
				}
			}
		}
	}
};

#endif
