#ifndef BACKWARD_H
#define BACKWARD_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;

namespace Backward
{
	void backward(auto w_lm, auto dz, auto H)
	{	
		auto dw = Tensor::dot_product(w_lm, dz);
	}
};

#endif