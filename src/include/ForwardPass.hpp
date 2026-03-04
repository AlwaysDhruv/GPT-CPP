#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <filesystem>

#include "Tensor.hpp"
#include "Display.hpp"
#include "Functions.hpp"
#include "Attension.hpp"
#include "ForwardPass.hpp"
#include "../utils/json.hpp"

using namespace std;
using json = nlohmann::json;
namespace fs = std::filesystem;
using ordered_json = nlohmann::ordered_json;

namespace Forward
{
	void Layers(auto& X_IN, auto& A, auto& H, auto& z, auto w_query, auto w_key, auto w_value, auto w_output, auto w1, auto w2, auto w_lm, auto b1, auto b2, auto b_lm, int head_size, int vocab_size, int layers)
	{
		for (size_t i = 0; i < layers; ++i)
		{
			auto X_NORM = Functions::normalizaion(X_IN);

			auto query_raw = Tensor::dot_product(X_NORM, w_query[i]);
			auto key_raw   = Tensor::dot_product(X_NORM, w_key[i]);
			auto values_raw = Tensor::dot_product(X_NORM, w_value[i]);			
			
			auto query = Tensor::reshape_to_multihead(query_raw, head_size);
			auto key = Tensor::reshape_to_multihead(key_raw, head_size);
			auto value = Tensor::reshape_to_multihead(values_raw, head_size);

			//Attension
			Multiheadattension block;
			
			auto attension = block.score(query, key, value);
			attension = Tensor::dot_product(attension, w_output[i]);

			auto X_NEW = Tensor::sum(attension, X_IN);

			//FFN
			X_NORM = Functions::normalizaion(X_NEW);

			auto Z = Tensor::dot_product(X_NORM, w1[i]);
			Tensor::sum(Z, b1[i]);

			z[i] = Z;
			
			Functions::gelu(Z);
			A[i] = Z;
		
			auto Z_FFN = Tensor::dot_product(Z, w2[i]);
			Tensor::sum(Z_FFN, b2[i]);

			X_IN = Tensor::sum(Z_FFN, X_NEW);
		}
		
		X_IN = Functions::normalizaion(X_IN);

		H = X_IN;

		X_IN = Tensor::dot_product(X_IN, w_lm);
		Tensor::sum(X_IN, b_lm);
		
		Functions::softmax(X_IN);
	}
};
#endif