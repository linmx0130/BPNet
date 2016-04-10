#include "cornernn_math.h"
namespace cornernn {
	template <>
	double sigmoid(double v) {
		return 1 / (1 + exp(-v));
	};
	template <>
	float sigmoid(float v) {
		return 1 / (1 + expf(-v));
	};
	template <>
	double tanh(double v) {
		return std::tanh(v);
	};
	template <>
	float tanh(float v) {
		return tanhf(v);
	};
}