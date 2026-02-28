#ifndef LAYERS_H
#define LAYERS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "Display.hpp"


namespace Layer
{
	float mean(vector<float> v)
	{
		float sum = 0.0f;
		for (size_t i = 0; i < v.size(); ++i) sum += v[i];
		
		return sum / v.size();
	}

	float variance(vector<float> v, float men)
	{
		float sum = 0.0f;
		for (size_t i = 0; i < v.size(); ++i) sum += ((v[i] - men) * (v[i] - men));
		
		return sum / v.size();
	}

	vector<vector<float>> normalizaion(vector<vector<float>> X_IN)
	{
		vector<vector<float>> result(X_IN.size(), vector<float>(X_IN[0].size(), 0.0f));
		for (size_t i = 0; i < X_IN.size(); ++i)
		{
			float men = mean(X_IN[i]);
			float var = variance(X_IN[i], men);
			
			for (size_t j = 0; j < X_IN[i].size(); ++j) result[i][j] = ((X_IN[i][j] - men) / sqrt(var));
		}
		return result;
	}
	
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