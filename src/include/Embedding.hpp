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
public:

	vector<vector<float>> init_weights(int vocab_size, int embed_size)
    {
        vector<vector<float>> weights(vocab_size, vector<float>(embed_size, 0.0f));
        random_device rd;
        mt19937 gen(rd());
        
        float limit = std::sqrt(6.0f / (vocab_size + embed_size));
        uniform_real_distribution<float> dist(-limit, limit);

        for (size_t i = 0; i < vocab_size; ++i) for (size_t j = 0; j < embed_size; ++j) weights[i][j] = dist(gen);

        return weights;
    }

    vector<vector<float>> forward(const vector<long long>& token_ids, const vector<vector<float>>& weights)
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
    	
    void positioning_encoding(vector<vector<float>>& embedding, const vector<vector<float>>& pos_weights)
    {
        for (size_t i = 0; i < embedding.size(); ++i)
        {
            for (size_t j = 0; j < embedding[i].size(); ++j)
            {
                embedding[i][j] += pos_weights[i][j];
            }
        }
    }
};

#endif