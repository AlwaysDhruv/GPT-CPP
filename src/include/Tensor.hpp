#ifndef TENSOR
#define TENSOR

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "Display.hpp"

using namespace std;

namespace Tensor
{
	vector<vector<float>> projection(vector<vector<float>>& vector1, vector<vector<float>>& vector2)
	{
		vector<vector<float>> vector3(vector2.size(), vector<float>(vector1[0].size(), 0.0f));
		for (size_t k = 0; k < vector2.size(); ++k)
			for (size_t i = 0; i < vector1.size(); ++i)
				for (size_t j = 0; j < vector1[0].size(); ++j)
					vector3[k][j] += vector1[j][i] * vector2[k][j];
		
		return vector3;
	}

	vector<vector<vector<float>>> reshape_to_multihead(vector<vector<float>>& vectorr, int head_size)
	{
		int ctc = 0;
		int dim_size = vectorr[0].size() / head_size;
		vector<vector<vector<float>>> values(head_size,vector<vector<float>>(vectorr.size(),vector<float>(dim_size)));
		for (size_t k = 0; k < head_size; ++k) for (size_t i = 0; i < vectorr.size(); ++i) for (size_t j = 0; j < dim_size; ++j) values[k][i][j] = vectorr[i][k * dim_size + j];
		
		return values;
	}
	
	vector<vector<float>> reshape_to_singlehead(vector<vector<vector<float>>>& multihead_vector)
	{
    	int head_size = multihead_vector.size();
    	int seq_len = multihead_vector[0].size();
    	int head_dim = multihead_vector[0][0].size();
    	int total_dim = head_size * head_dim;
	    vector<vector<float>> single_matrix(seq_len, vector<float>(total_dim));

    	for (int k = 0; k < head_size; ++k)for (int i = 0; i < seq_len; ++i) for (int j = 0; j < head_dim; ++j) single_matrix[i][k * head_dim + j] = multihead_vector[k][i][j];

    	return single_matrix;
	}

	vector<vector<float>> transpose(vector<vector<float>>& vectorr)
	{
		vector<vector<float>> values(vectorr[0].size(), vector<float>(vectorr.size(), 0.0f));
		for (size_t i = 0; i < vectorr[0].size(); ++i) for (size_t j = 0; j < vectorr.size(); ++j) values[i][j] = vectorr[j][i];
			
		return values;
	}

   	vector<vector<float>> dot_product(vector<vector<float>>& vector1, vector<vector<float>>& vector2)
	{
		vector<vector<float>> vector3(vector1.size(), vector<float>(vector2[0].size(), 0.0f));

		for (size_t i = 0; i < vector1.size(); ++i)
		{
			for (size_t j = 0; j < vector2[0].size(); ++j)
			{
				float sum = 0.0f;
				for (size_t k = 0; k < vector2.size(); ++k) sum += vector1[i][k] * vector2[k][j];
				vector3[i][j] = sum;
			}
		}
		return vector3;
	}
   	
	void concate(vector<vector<vector<float>>>& vector1, vector<vector<float>>& vector2)
	{
		int ct = 0;
		for (size_t i = 0; i < vector1.size(); ++i)
		{
			for (size_t j = 0; j < vector1[0].size(); ++j) for (size_t l = 0; l < vector1[0][0].size(); ++l) vector2[j][l + ct] = vector1[i][j][l];
			ct += vector1[0][0].size();
		}
	}
	
	vector<vector<float>> sum(vector<vector<float>>& vector1, vector<vector<float>>& vector2)
	{
		vector<vector<float>> sumation(vector1.size(), vector<float>(vector2[0].size()));
		for (size_t i = 0; i < vector1.size(); ++i) for (size_t j = 0; j < vector2[0].size(); ++j) sumation[i][j] = vector1[i][j] + vector2[i][j];

		return sumation;
	}
	
	vector<vector<float>> sub(vector<vector<float>>& vector1, vector<vector<float>>& vector2)
	{
		vector<vector<float>> sumation(vector1.size(), vector<float>(vector2[0].size()));
		for (size_t i = 0; i < vector1.size(); ++i) for (size_t j = 0; j < vector2[0].size(); ++j) sumation[i][j] = vector1[i][j] - vector2[i][j];

		return sumation;
	}
	
	void sum(vector<vector<float>>& vector1, vector<float>& vector2)
	{
		for (size_t i = 0; i < vector1.size(); ++i) for (size_t j = 0; j < vector2.size(); ++j) vector1[i][j] = vector1[i][j] + vector2[j];
	}

	vector<float> bias(vector<vector<float>>& dz)
	{
		vector<float> b(dz[0].size());
		for (int i = 0; i < dz.size(); ++i) for (int j = 0; j < dz[0].size(); ++j) b[j] += dz[i][j];

		return b;
	}

	vector<vector<float>> multiply(vector<vector<float>> vector1, vector<vector<float>> vector2)
	{
		vector<vector<float>> vector3(vector1.size(), vector<float>(vector2[0].size(), 0.0f));
		for (size_t t = 0; t < vector1.size(); ++t) for (size_t j = 0; j < vector1[0].size(); ++j) vector3[t][j] = vector1[t][j] * vector2[t][j];

		return vector3;
	}
};
#endif