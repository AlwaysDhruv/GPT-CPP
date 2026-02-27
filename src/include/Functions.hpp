#ifndef FUNCTIONS
#define FUNCTIONS

#include <iostream>
#include <vector>
#include <cmath>

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
}

#endif