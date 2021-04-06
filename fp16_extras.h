#include "fp16_emu.h"

std::ostream& operator<<(std::ostream& os, half1& arg) {
    return os << __half2float(arg);
}

std::istream& operator>>(std::istream& is, half1& arg)
{
    double f;
	if(is >> f)
		arg = __float2half(f);
	return is;
}

__device__ half1 operator+(half1& a, half1& b) {
    return __hadd(a, b);
}

__device__ half1 operator-(half1& a, half1& b) {
    return __hsub(a, b);
}

__device__ half1 operator*(half1& a, half1& b) {
    return __hmul(a, b);
}

__device__ half1 operator/(half1& a, half1& b) {
    return __hdiv(a, b);
}