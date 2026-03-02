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
	vector<vector<float>> Layers(auto& X_IN, auto w_query, auto w_key, auto w_value, auto w_output, auto w1, auto w2, auto w_lm, auto b1, auto b2, auto b_lm, int head_size, int vocab_size, int layers)
	{
		for (size_t i = 0; i < layers; ++i)
		{
			auto X_NORM = Functions::normalizaion(X_IN);

			auto query_raw = Tensor::projection(w_query[i], X_NORM);
			auto key_raw = Tensor::projection(w_key[i], X_NORM);
			auto values_raw = Tensor::projection(w_value[i], X_NORM);
			
			auto query = Tensor::reshape_to_multihead(query_raw, head_size);
			auto key = Tensor::reshape_to_multihead(key_raw, head_size);
			auto value = Tensor::reshape_to_multihead(values_raw, head_size);

			//Attension
			Multiheadattension block;
			
			auto Z = block.score(query, key, value);
			Z = Tensor::dot_product(Z, w_output[i]);

			auto X_NEW = Tensor::sum(Z, X_IN);

			//FFN
			X_NORM = Functions::normalizaion(X_NEW);

			auto H = Tensor::dot_product(X_NORM, w1[i]);
			Tensor::sum(H, b1[i]);

			Functions::gelu(H);
		
			auto Z_FFN = Tensor::dot_product(H, w2[i]);
			Tensor::sum(Z_FFN, b2[i]);

			X_IN = Tensor::sum(Z_FFN, X_NEW);
		}
		
		X_IN = Functions::normalizaion(X_IN);

		auto H = X_IN;

		X_IN = Tensor::dot_product(X_IN, w_lm);
		Tensor::sum(X_IN, b_lm);
		
		Functions::softmax(X_IN);
		return H;
	}
};
#endif