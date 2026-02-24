#ifndef LINEAR_PROJECTION
#define LINEAR_PROJECTION

#include <iostream>
#include <random>
#include <vector>
#include <cmath>

using namespace std;

class Linear
{
public:
	vector<vector<float>> linear(int cols_size, int rows_size)
	{
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<float> dist(-1.0, 1.0);
		vector<vector<float>>values(cols_size, vector<float>(rows_size, 0.0f));
		for (size_t j = 0; j < cols_size; ++j) for (size_t k = 0; k < rows_size; ++k) values[j][k] = round(dist(gen) * 100) / 100;
		return values;
	}
	vector<vector<float>> linear(int dim_size)
	{
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<float> dist(-1.0, 1.0);
		vector<vector<float>>values(dim_size, vector<float>(dim_size, 0.0f));
		for (size_t j = 0; j < dim_size; ++j) for (size_t k = 0; k < dim_size; ++k) values[j][k] = round(dist(gen) * 100) / 100;
		return values;
	}

};

#endif