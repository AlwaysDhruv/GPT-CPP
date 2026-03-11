#ifndef ATTENSION
#define ATTENSION

#include <iostream>
#include <vector>
#include <cmath>
#include "Tensor.hpp"
#include "Display.hpp"

using namespace std;

class Multiheadattension
{
public:
	vector<vector<float>> score(vector<vector<vector<float>>> query, vector<vector<vector<float>>> key, vector<vector<vector<float>>> value, auto& score)
	{	
		int head_size = query.size();
		int seq_len = query[0].size();
		int v_dim = value[0][0].size();
		
		float scale = 1.0f / sqrt(static_cast<float>(v_dim));

		vector<vector<vector<float>>> A(head_size, vector<vector<float>>(seq_len, vector<float>(v_dim)));

		for (size_t j = 0; j < head_size; ++j)
		{
			vector<vector<float>> kT = Tensor::transpose(key[j]);
			
			vector<vector<float>>attension_score = Tensor::dot_product(query[j], kT);
			
			Functions::scaling(attension_score, scale);

			Functions::causal_mask(attension_score);
			
			Functions::softmax(attension_score);
			
			score[j] = attension_score;
			
			A[j] = Tensor::dot_product(attension_score, value[j]);
		}

		vector<vector<float>> Z(seq_len, vector<float>(head_size * v_dim));
		Tensor::concate(A, Z);
		return Z;
	}
	
	vector<vector<float>> score(vector<vector<vector<float>>> query, vector<vector<vector<float>>> key, vector<vector<vector<float>>> value)
	{	
		int head_size = query.size();
		int seq_len = query[0].size();
		int v_dim = value[0][0].size();

		float scale = sqrt(static_cast<float>(v_dim));

		vector<vector<vector<float>>> A(head_size, vector<vector<float>>(seq_len, vector<float>(v_dim)));

		for (size_t j = 0; j < head_size; ++j)
		{
			vector<vector<float>> kT = Tensor::transpose(key[j]);
			
			vector<vector<float>>attension_score = Tensor::dot_product(query[j], kT);
			
			Functions::causal_mask(attension_score);
			
			Functions::softmax(attension_score);
			
			A[j] = Tensor::dot_product(attension_score, value[j]);
		}

		vector<vector<float>> Z(seq_len, vector<float>(head_size * v_dim));
		Tensor::concate(A, Z);
		return Z;
	}
};
#endif