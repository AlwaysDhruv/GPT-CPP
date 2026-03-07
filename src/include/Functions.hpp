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

    float gelu_der(float x)
    {
    	const float kSqrt2OverPi = 0.79788456f;
    	const float kCoeff = 0.044715f;
    
    	float x2 = x * x;
    	float x3 = x * x2;
    
    	float inner = kSqrt2OverPi * (x + kCoeff * x3);
    	float tanh_val = std::tanh(inner);
    
    	float left_side = 0.5f * (1.0f + tanh_val);
    	float right_side = 0.5f * x * (1.0f - tanh_val * tanh_val) * kSqrt2OverPi * (1.0f + 3.0f * kCoeff * x2);
    
    	return left_side + right_side;
	}

    void gelu(vector<vector<float>>& H)
    {
        int rows = H.size();
        int cols = H[0].size();

        for (size_t i = 0; i < rows; ++i) for (size_t j = 0; j < cols; ++j) H[i][j] = gelu_single(H[i][j]);
    }

    void gelu_derivative(vector<vector<float>>& H)
    {
        int rows = H.size();
        int cols = H[0].size();

        for (size_t i = 0; i < rows; ++i) for (size_t j = 0; j < cols; ++j) H[i][j] = gelu_der(H[i][j]);
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
    	const float eps = 1e-5f; // Small constant to prevent division by zero
    	vector<vector<float>> result(X_IN.size(), vector<float>(X_IN[0].size(), 0.0f));
    
    	for (size_t i = 0; i < X_IN.size(); ++i)
    	{
        	float men = mean(X_IN[i]);
        	float var = variance(X_IN[i], men);
        
        	float std_dev = sqrt(var + eps);
        
        	for (size_t j = 0; j < X_IN[i].size(); ++j) result[i][j] = (X_IN[i][j] - men) / std_dev;
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

	vector<vector<vector<vector<float>>>> linear3(int layers, int head_size, int rows, int cols)
	{
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<float> dist(-1.0, 1.0);
		vector<vector<vector<vector<float>>>>values(layers, vector<vector<vector<float>>>(head_size, vector<vector<float>>(rows, vector<float>(cols, 0.0f))));
		for (size_t i = 0; i < layers; ++i) for (size_t j = 0; j < head_size; ++j) for (size_t k = 0; k < rows; ++k) for (size_t l = 0; l < cols; ++l) values[i][j][k][l] = round(dist(gen) * 100) / 100;
		
		return values;

	}
	vector<vector<float>> linear2(int cols_size, int rows_size)
	{
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<float> dist(-1.0, 1.0);
		vector<vector<float>>values(cols_size, vector<float>(rows_size, 0.0f));
		for (size_t j = 0; j < cols_size; ++j) for (size_t k = 0; k < rows_size; ++k) values[j][k] = round(dist(gen) * 100) / 100;
		
		return values;
	}

	vector<long long> target_shift(vector<long long> token_ids)
	{
		vector<long long> target_ids(token_ids.size());
		for (size_t i = 0; i < token_ids.size(); ++i) target_ids[i] = token_ids[i];

		return target_ids;
	}

	vector<vector<float>> gradient_loss(auto forward, auto Y)
	{
		float loss = 0.0f;
		auto dz = forward;
		auto seq_len = forward.size();
		for (size_t i = 0; i < forward.size(); ++i)
		{
			auto target_id = Y[i];
			
			loss += -log(forward[i][target_id] + 1e-9f);
			dz[i][target_id] -= 1.0f;
			
			for (size_t j = 0; j < forward[i].size(); ++j) dz[i][j] /= (float)seq_len;
		}
		return dz;
	}
	
	void update(vector<vector<float>>& w, vector<vector<float>> dw, float rate)
	{
		for (size_t i = 0; i < w.size(); ++i) for (size_t j = 0; j < w[0].size(); ++j) w[i][j] -= rate * dw[i][j];
	}
	
	void update(vector<vector<vector<float>>>& w, vector<vector<vector<float>>>& dw, float rate)
	{
    	for (size_t h = 0; h < w.size(); ++h) for (size_t i = 0; i < w[h].size(); ++i) for (size_t j = 0; j < w[h][0].size(); ++j) w[h][i][j] -= rate * dw[h][i][j];
	}

	void update(vector<float>& b, vector<float> db, float rate)
	{
		for (size_t i = 0; i < b.size(); ++i) b[i] -= rate * db[i];
	}	
}
#endif