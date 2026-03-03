#ifndef BACKWARD_H
#define BACKWARD_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;

namespace Backward
{
	void update(vector<vector<float>>& w, vector<vector<float>> dw, float rate)
	{
		for (size_t i = 0; i < w.size(); ++i) for (size_t j = 0; j < w[0].size(); ++j) w[i][j] -= rate * dw[i][j];
	}
	
	void update(vector<float>& b, vector<float> db, float rate)
	{
		for (size_t i = 0; i < b.size(); ++i) b[i] -= rate * db[i];
	}

	void backward(auto w_lm, auto b_lm, auto dz, auto H, auto rate)
	{
		auto seq_len = dz.size();
    	auto vocab_size = dz[0].size();
    	auto hidden_dim = H[0].size();

	    vector<vector<float>> dw_lm(hidden_dim, vector<float>(vocab_size, 0.0f));
	    vector<float> db_lm(vocab_size, 0.0f);

    	for (size_t t = 0; t < seq_len; ++t)
    		for (size_t i = 0; i < hidden_dim; ++i)
    			for (size_t j = 0; j < vocab_size; ++j)
    			{
    				dw_lm[i][j] += H[t][i] * dz[t][j];
    				db_lm[j] += dz[t][j];
    			}
		
		vector<vector<float>> dh(seq_len, vector<float>(hidden_dim, 0.0f));

		for (size_t i = 0; i < dz.size(); ++i)
		{
			for (size_t j = 0; j < w_lm.size(); ++j)
			{
				float sum = 0.0;
				for (size_t t = 0; t < w_lm[0].size(); ++t) sum += dz[j][t] * w_lm[j][t];
				dh[i][j] = sum;
			}
		}
		update(w_lm, dw_lm, rate);
		update(b_lm, db_lm, rate);
	}
};
#endif