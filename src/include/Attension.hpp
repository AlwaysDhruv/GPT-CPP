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
	void decoder(vector<vector<vector<float>>> query, vector<vector<vector<float>>> key, vector<vector<vector<float>>> value)
	{
		Tensor tnsr;
		
		float scale = 1.0f / sqrt(key.size());
		int head_size = query.size();
		for (size_t j = 0; j < head_size; ++j)
		{
			vector<vector<float>> attension_score(query[j].size(),vector<float>(key[j].size()));
			
			key[j] = tnsr.transpose(key[j]);
			
			attension_score = tnsr.dot_product2(query[j], key[j]);
			
			Functions::casual_mask(attension_score, scale);
			Functions::softmax(attension_score);
			
			vector<vector<float>> A(attension_score.size(),vector<float>(value[j].size()));
			
			A = tnsr.dot_product2(attension_score, value[j]);
		}
	}
};
#endif