#ifndef ATTENSION
#define ATTENSION

#include <iostream>
#include <vector>
#include <cmath>
#include "Tensor.hpp"
#include "Display.hpp"
using namespace std;

class Attension
{
public:
	void scores(vector<vector<vector<float>>> query, vector<vector<vector<float>>> key, vector<vector<vector<float>>> value)
	{
		Tensor tnsr;
		
		float scale = 1.0f / sqrt(key.size());

		for (size_t j = 0; j < query.size(); ++j)
		{
			vector<vector<float>> attension_score(query[0].size(),vector<float>(key[j].size()));
			
			key[j] = tnsr.transpose(key[j]);

			for (size_t i = 0; i < query[0].size(); ++i)
			{
				for (size_t k = 0; k < key[0][0].size(); ++k)
				{
					float sum = 0.0f;
					for (size_t l = 0; l < key[0].size(); ++l) sum += query[j][i][l] * key[j][l][k];
					attension_score[i][k] = sum;
				}
			}

			for (size_t i = 0; i < attension_score.size(); ++i)
			{
				for (size_t k = 0; k < attension_score[0].size(); ++k)
				{
					if (k > i) attension_score[i][k] = -1e9f;
					else attension_score[i][k] *= scale;
				}
			}			
			softmax(attension_score);
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
};
#endif