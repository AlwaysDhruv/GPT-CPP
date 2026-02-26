#ifndef TENSOR
#define TENSOR

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "Display.hpp"

using namespace std;

class Tensor
{
public:
	vector<vector<float>> dot_product(vector<vector<float>>& vector1, vector<vector<float>>& vector2)
	{
		vector<vector<float>> vector3(vector2.size(), vector<float>(vector1[0].size(), 0.0f));
		for (size_t k = 0; k < vector2.size(); ++k) for (size_t i = 0; i < vector1.size(); ++i) for (size_t j = 0; j < vector1[0].size(); ++j) vector3[k][j] += vector1[j][i] * vector2[k][j];
		
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

	vector<vector<float>> transpose(vector<vector<float>>& vectorr)
	{
		vector<vector<float>> values(vectorr[0].size(), vector<float>(vectorr.size(), 0.0f));
		for (size_t i = 0; i < vectorr[0].size(); ++i) for (size_t j = 0; j < vectorr.size(); ++j) values[i][j] = vectorr[j][i];
			
		return values;
	}
};
#endif