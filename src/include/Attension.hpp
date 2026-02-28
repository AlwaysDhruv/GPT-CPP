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
	vector<vector<float>> decoder(vector<vector<vector<float>>> query, vector<vector<vector<float>>> key, vector<vector<vector<float>>> value)
	{
		Tensor tnsr;
		
		int head_size = query.size();
		int seq_len = query[0].size();
		int v_dim = value[0][0].size();

		float scale = 1.0f / sqrt(static_cast<float>(v_dim));

		vector<vector<vector<float>>> A(head_size, vector<vector<float>>(seq_len, vector<float>(v_dim)));

		for (size_t j = 0; j < head_size; ++j)
		{
			vector<vector<float>> kT = tnsr.transpose(key[j]);
			
			vector<vector<float>>attension_score = tnsr.dot_product(query[j], kT);
			
			Functions::casual_mask(attension_score, scale);
			
			Functions::softmax(attension_score);
			
			A[j] = tnsr.dot_product(attension_score, value[j]);
		}

		vector<vector<float>> Z(seq_len, vector<float>(head_size * v_dim));
		tnsr.concate(A, Z);

		return Z;
	}

};
#endif