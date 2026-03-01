#ifndef FUNCTIONS
#define FUNCTIONS

#include <iostream>
#include <vector>
#include <cmath>
#include <numbers>

namespace Functions
{
	void casual_mask(vector<vector<float>>& attension_score, float scale)
	{
		for (size_t i = 0; i < attension_score.size(); ++i)
		{
			for (size_t k = 0; k < attension_score[0].size(); ++k)
			{
				if (k > i) attension_score[i][k] = -1e9f;
				else attension_score[i][k] *= scale;
			}
		}
	}
	
	void softmax(vector<vector<float>>& matrix)
	{
    	int rows = matrix.size();
    	if (rows == 0) return;
    	int cols = matrix[0].size();

    	for (size_t i = 0; i < rows; ++i)
    	{
        	float max_val = *max_element(matrix[i].begin(), matrix[i].end());
	        float sum = 0.0f;
    	    
    	    for (size_t j = 0; j < cols; ++j)
    	    {
            	matrix[i][j] = exp(matrix[i][j] - max_val);
            	sum += matrix[i][j];
        	}

	        for (size_t j = 0; j < cols; ++j)
	        {
            	if (sum > 0.0f) matrix[i][j] /= sum;
            	else matrix[i][j] = 0.0f;
            }
        }
    }
	
	float gelu_single(float x)
	{
        const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
        const float coeff = 0.044715f;

        float x_cubed = std::pow(x, 3);
        float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        
        return 0.5f * x * (1.0f + tanh(inner));
    }

    void gelu(vector<vector<float>>& H)
    {
        int rows = H.size();
        int cols = H[0].size();

        for (size_t i = 0; i < rows; ++i) for (size_t j = 0; j < cols; ++j) H[i][j] = gelu_single(H[i][j]);
    }
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
	
	vector<vector<vector<float>>> linear(int layers, int cols_size, int rows_size)
	{
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<float> dist(-1.0, 1.0);
		vector<vector<vector<float>>>values(layers, vector<vector<float>>(cols_size, vector<float>(rows_size, 0.0f)));
		for (size_t i = 0; i < layers; ++i) for (size_t j = 0; j < cols_size; ++j) for (size_t k = 0; k < rows_size; ++k) values[i][j][k] = round(dist(gen) * 100) / 100;
		
		return values;
	}
	
	vector<vector<vector<float>>> linear(int layers, int dim_size)
	{
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<float> dist(-1.0, 1.0);
		vector<vector<vector<float>>>values(layers, vector<vector<float>>(dim_size, vector<float>(dim_size, 0.0f)));
		for (size_t i = 0; i < layers; ++i) for (size_t j = 0; j < dim_size; ++j) for (size_t k = 0; k < dim_size; ++k) values[i][j][k] = round(dist(gen) * 100) / 100;
		
		return values;
	}
}
#endif