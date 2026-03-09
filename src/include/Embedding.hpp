#ifndef VALUES_H
#define VALUES_H

#include <iostream>
#include <random>
#include <vector>
#include <cmath>

#include "Display.hpp"

using namespace std;

class Embedd
{
	vector<vector<float>> weights;
public:

	void init_weights(int vocab_size, int embed_size)
    {
        weights.assign(vocab_size, vector<float>(embed_size, 0.0f));
        random_device rd;
        mt19937 gen(rd());
        
        // Glorot/Xavier Initialization uses Vocab Size, not Seq Len
        float limit = std::sqrt(6.0f / (vocab_size + embed_size));
        uniform_real_distribution<float> dist(-limit, limit);

        for (size_t i = 0; i < vocab_size; ++i) {
            for (size_t j = 0; j < embed_size; ++j) {
                weights[i][j] = dist(gen);
            }
        }
    }

    vector<vector<float>> forward(const vector<long long>& token_ids)
    {
        int seq_len = token_ids.size();
        int embed_size = weights[0].size();
        vector<vector<float>> values(seq_len, vector<float>(embed_size, 0.0f));
        for (size_t i = 0; i < seq_len; ++i)
        {
            long long token_id = token_ids[i];
            for (size_t j = 0; j < embed_size; ++j) {
                values[i][j] = weights[token_id][j];
            }
        }
        return values;
    }
    	
    void positioning_encoding(vector<vector<float>>& embedding)
	{
		int dim = embedding[0].size();
		
		for (size_t i = 0; i < embedding.size(); ++i)
		{
			float values;
			for (size_t j = 0; j < embedding[i].size(); ++j)
			{
				if (j % 2 == 0) embedding[i][j] = embedding[i][j] + sin(i / pow(10000.0, j / (float)dim));
				else embedding[i][j] = embedding[i][j] + cos(i / pow(10000.0, (j - 1) / (float)dim));
			}
		}
	}
};

#endif