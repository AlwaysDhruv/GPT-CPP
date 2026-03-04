#ifndef BACKWARD_H
#define BACKWARD_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;

namespace Backward
{
	vector<vector<float>> backward_LM(auto& w_lm, auto& b_lm, auto H, auto dz, auto rate)
	{
		auto seq_len = dz.size();
    	auto vocab_size = dz[0].size();
    	auto hidden_dim = H[0].size();

    	H = Tensor::transpose(H);
    	
    	auto dw_lm = Tensor::dot_product(H, dz);
    	auto db_lm = Tensor::bias(dz);
    	
    	vector<vector<float>> dh(seq_len, vector<float>(hidden_dim, 0.0f));
		for (size_t t = 0; t < seq_len; ++t)
		{
			for (size_t i = 0; i < hidden_dim; ++i)
			{
				float sum = 0.0;
				for (size_t j = 0; j < vocab_size; ++j) sum += dz[t][j] * w_lm[i][j];
				dh[t][i] = sum;
			}
		}
		
		Functions::update(w_lm, dw_lm, rate);
		Functions::update(b_lm, db_lm, rate);
		
		return dh;
	}

	vector<vector<float>> backward_FFN(auto& w1, auto& w2, auto& b1, auto& b2, auto A, auto Z, auto X, auto dh, auto rate)
	{
		auto layers = w2.size();

		for (int i = layers - 1; i >= 0; --i)
		{
			A[i] = Tensor::transpose(A[i]);
			auto dw2 = Tensor::dot_product(A[i], dh);
			auto db2 = Tensor::bias(dh);

			auto w2i = Tensor::transpose(w2[i]);
			dh = Tensor::dot_product(dh, w2i);
			
			Functions::gelu_derivative(Z[i]);

			auto dz = Tensor::dot_product(dh, Z[i]);

			X = Tensor::transpose(X);
			auto dw1 = Tensor::dot_product(X, dz);
			auto db1 = Tensor::bias(dz);

			auto w1i = Tensor::transpose(w1[i]);
			dh = Tensor::dot_product(dz, w1i);

			Functions::update(w2[i], dw2, rate);
			Functions::update(w1[i], dw1, rate);

			Functions::update(b2[i], db2, rate);
			Functions::update(b1[i], db1, rate);
		}
		return dh;
	}
};
#endif