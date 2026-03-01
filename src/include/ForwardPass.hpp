#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "Tensor.hpp"
#include "Display.hpp"
#include "Functions.hpp"
#include "Attension.hpp"
#include "ForwardPass.hpp"

namespace Forward
{
	vector<vector<float>> Layers(auto query, auto key, auto value, auto w_output, auto w1, auto w2, auto b1, auto b2, auto X_IN, auto X_NORM, int layer)
	{
		Tensor tensr;
		
		for (size_t i = 0; i < layer; ++i)
		{
			//Attension
			Multiheadattension block;
			auto Z = block.score(query, key, value);
			Z = tensr.dot_product(Z, w_output);
			auto X_NEW = tensr.sum(Z, X_IN);
			
			//FFN
			X_NORM = Functions::normalizaion(X_NEW);
			auto H = tensr.dot_product(X_NORM, w1);
			tensr.sum(H, b1);
			Functions::gelu(H);
			
			auto Z_FFN = tensr.dot_product(H, w2);
			tensr.sum(Z_FFN, b2);
			X_IN = tensr.sum(Z_FFN, X_NEW);
		}

		return X_IN;
	}
};
#endif