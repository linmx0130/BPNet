#ifndef CORNERNN_MATH_H
#define CORNERNN_MATH_H
#include "stdc.h"

namespace cornernn {
	template <typename FLOAT_TYPE> FLOAT_TYPE sigmoid(FLOAT_TYPE v) {
		return 1 / (1 + exp((double) (-v)));
	};
	template <>
	double sigmoid(double v) {
		return 1 / (1 + exp(-v));
	};
	template <>
	float sigmoid(float v) {
		return 1 / (1 + expf(-v));
	};
	template <typename FLOAT_TYPE> FLOAT_TYPE derivSigmoid(FLOAT_TYPE v) {
		return v * (1.0 - v);
	};

	template <typename FLOAT_TYPE> FLOAT_TYPE tanh(FLOAT_TYPE v) {
		return tanh((double)v);
	};
	template <>
	double tanh(double v) {
		return std::tanh(v);
	};
	template <>
	float tanh(float v) {
		return tanhf(v);
	};
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
					d[i][j] = ((FLOAT_TYPE)(rand() % 10000)) / 10000 /divisor;
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
		
		Matrix<FLOAT_TYPE, ROW, COLUMN> &operator*= (FLOAT_TYPE v) {
			for (int i = 0; i < ROW; ++i) {
				for (int j = 0; j < COLUMN; ++j) {
					d[i][j] *= v;
				}
			}
			return *this;
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
};

#endif
