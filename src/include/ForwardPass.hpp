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
#include "../utils/ini.h"
using namespace std;
using json = nlohmann::json;
namespace fs = std::filesystem;
using ordered_json = nlohmann::ordered_json;

namespace Forward
{

	vector<vector<float>> Layers(mINI::INIStructure& ini, auto X_IN, auto& A, auto& H, auto& z, auto& at , auto& score, const auto& w_query, const auto& w_key, const auto& w_value, const auto& w_output, const auto& w1, const auto& w2, auto embbed_matrix, const auto& b1, const auto& b2, const auto& b_lm, auto& qry, auto& ky, auto& val, auto& X_IN2, auto& X_IN3)
	{
		int embed_size = stoi(ini["GPT"]["Emdedding_size"]);
		int head_size = stoi(ini["GPT"]["Head_size"]);
		int layers = stoi(ini["GPT"]["Layers"]);
		int vocab_size = stoi(ini["GPT"]["Vocab_size"]);
		float learning_rate = stof(ini["GPT"]["Rate"]);
		int embed_size2 = embed_size * embed_size;
		int seq_len = X_IN.size();

		for (size_t i = 0; i < layers; ++i)
		{
			auto X_NORM = Functions::normalizaion(X_IN);

			X_IN3[i] = X_NORM;
			
			vector<vector<vector<float>>> query(head_size ,vector<vector<float>>(seq_len,vector<float>(embed_size / head_size)));
			vector<vector<vector<float>>> key(head_size ,vector<vector<float>>(seq_len,vector<float>(embed_size / head_size)));
			vector<vector<vector<float>>> value(head_size ,vector<vector<float>>(seq_len,vector<float>(embed_size / head_size)));			
			
			for (int j = 0; j < head_size; ++j)
			{
    			query[j] = Tensor::dot_product(X_NORM, w_query[i][j]);
    			key[j]   = Tensor::dot_product(X_NORM, w_key[i][j]);
    			value[j] = Tensor::dot_product(X_NORM, w_value[i][j]);
			}

			qry[i] = query;
			ky[i] = key;
			val[i] = value;

			//Attension
			Multiheadattension block;
			
			auto attension = block.score(query, key, value, score[i]);
			
			attension = Tensor::dot_product(attension, w_output[i]);

			at[i] = attension;

			auto X_NEW = Tensor::sum(attension, X_IN);

			//FFN
			X_NORM = Functions::normalizaion(X_NEW);
			
			X_IN2[i] = X_NORM;
			
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

		vector<vector<float>> embbed_T = Tensor::transpose(embbed_matrix);
		
		X_IN = Tensor::dot_product(X_IN, embbed_T);
		Tensor::sum(X_IN, b_lm);
		
		Functions::softmax(X_IN);

		return X_IN;
	}

	vector<vector<float>> predict(mINI::INIStructure& ini,
		auto X_IN, const auto& w_query,
		const auto& w_key, const auto& w_value,
		const auto& w_output, const auto& w1,
		const auto& w2, auto embbed_matrix, const auto& b1, const auto& b2, const auto& b_lm)
	{
		int embed_size = stoi(ini["GPT"]["Emdedding_size"]);
		int head_size = stoi(ini["GPT"]["Head_size"]);
		int layers = stoi(ini["GPT"]["Layers"]);
		int vocab_size = stoi(ini["GPT"]["Vocab_size"]);
		float learning_rate = stof(ini["GPT"]["Rate"]);
		int embed_size2 = embed_size * embed_size;
		int seq_len = X_IN.size();

		for (size_t i = 0; i < layers; ++i)
		{
			auto X_NORM = Functions::normalizaion(X_IN);
			
			vector<vector<vector<float>>> query(head_size ,vector<vector<float>>(seq_len,vector<float>(embed_size / head_size)));
			vector<vector<vector<float>>> key(head_size ,vector<vector<float>>(seq_len,vector<float>(embed_size / head_size)));
			vector<vector<vector<float>>> value(head_size ,vector<vector<float>>(seq_len,vector<float>(embed_size / head_size)));			
			
			for (int j = 0; j < head_size; ++j)
			{
    			query[j] = Tensor::dot_product(X_NORM, w_query[i][j]);
    			key[j]   = Tensor::dot_product(X_NORM, w_key[i][j]);
    			value[j] = Tensor::dot_product(X_NORM, w_value[i][j]);
			}

			//Attension
			Multiheadattension block;
			
			auto attension = block.score(query, key, value);
			
			attension = Tensor::dot_product(attension, w_output[i]);

			auto X_NEW = Tensor::sum(attension, X_IN);

			//FFN
			X_NORM = Functions::normalizaion(X_NEW);
			
			auto Z = Tensor::dot_product(X_NORM, w1[i]);
			Tensor::sum(Z, b1[i]);
			
			Functions::gelu(Z);
		
			auto Z_FFN = Tensor::dot_product(Z, w2[i]);
			Tensor::sum(Z_FFN, b2[i]);

			X_IN = Tensor::sum(Z_FFN, X_NEW);
		}
		
		X_IN = Functions::normalizaion(X_IN);

		vector<vector<float>> embbed_T = Tensor::transpose(embbed_matrix);
		
		X_IN = Tensor::dot_product(X_IN, embbed_matrix);
		Tensor::sum(X_IN, b_lm);
		
		Functions::softmax(X_IN);

		return X_IN;
	}	
};
#endif