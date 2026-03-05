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

	vector<vector<float>> backward_FFN(auto& w1, auto& w2, auto& b1, auto& b2, auto A, auto Z, auto AT, auto X, auto dh, auto rate)
	{
		auto layers = w2.size() - 1;

		for (int i = layers; i >= 0; --i)
		{
			A[i] = Tensor::transpose(A[i]);
			auto dw2 = Tensor::dot_product(A[i], dh);
			auto db2 = Tensor::bias(dh);

			auto w2i = Tensor::transpose(w2[i]);
			dh = Tensor::dot_product(dh, w2i);
			
			Functions::gelu_derivative(Z[i]);

			auto dz = dh;
			for (size_t t = 0; t < dz.size(); ++t) for (size_t j = 0; j < dz[0].size(); ++j) dz[t][j] = dz[t][j] * Z[i][t][j];

			X[i] = Tensor::transpose(X[i]);
			auto dw1 = Tensor::dot_product(X[i], dz);
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

	vector<vector<vector<vector<float>>>> backward_Attension1(auto& w_output, auto AT, auto dh, auto& da_h, auto score, auto head_size, auto rate)
	{
		auto layers = AT.size();
		vector<vector<vector<vector<float>>>> dv_h(layers, vector<vector<vector<float>>>(head_size, vector<vector<float>>(AT[0].size(),vector<float>(AT[0][0].size() / head_size, 0.0f))));

		for (int i = layers - 1; i >= 0; --i)
		{
			auto AT_T = Tensor::transpose(AT[i]);
			auto dw_output = Tensor::dot_product(AT_T, dh);

			auto w_output_t = Tensor::transpose(w_output[i]);
			auto da_concate = Tensor::dot_product(dh, w_output_t);
			
			auto da_concate_m = Tensor::reshape_to_multihead(da_concate, head_size);
			da_h[i] = da_concate_m;
			
			for (int j = 0; j < head_size; ++j)
			{
				auto score_t = Tensor::transpose(score[i][j]);
				dv_h[i][j] =  Tensor::dot_product(score_t, da_concate_m[j]);
			}
			Functions::update(w_output[i], dw_output, rate);
		}
		
		return dv_h;
	}

	void backward_Attension2(auto da_h, auto value, auto score)
	{
		int layers = da_h.size();
		int head_size = da_h[0].size();

		for (int i = layers - 1; i >= 0; --i)
		{
			for (int j = 0; j < head_size; ++j)
			{
				auto value_t = Tensor::transpose(value[i][j]);
				auto dscore_h = Tensor::dot_product(da_h[i][j],value_t);
				//Softmax Jacobian
			}
		}
	}
};
#endif