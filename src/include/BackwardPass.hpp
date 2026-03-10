#ifndef BACKWARD_H
#define BACKWARD_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;

namespace Backward
{
    void backward_LM(const auto& embbed_matrix, auto& b_lm, auto H, auto dz, auto rate, auto& grad_embed, auto& dh)
    {

        auto seq_len = dz.size();
        auto vocab_size = dz[0].size();
        auto hidden_dim = H[0].size();

        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t i = 0; i < hidden_dim; ++i) {
                float sum = 0.0;
                for (size_t j = 0; j < vocab_size; ++j) {
                    sum += dz[t][j] * embbed_matrix[j][i];
                }
                dh[t][i] = sum;
            }
        }

        auto dz_T = Tensor::transpose(dz);
        auto dw_lm_tied = Tensor::dot_product(dz_T, H);

        for(size_t i = 0; i < vocab_size; ++i) {
            for(size_t j = 0; j < hidden_dim; ++j) {
                grad_embed[i][j] += dw_lm_tied[i][j];
            }
        }

        auto db_lm = Tensor::bias(dz);
        Functions::update(b_lm, db_lm, rate);
    }
	vector<vector<float>> backward_Transformer(auto& w1, auto& w2, auto& b1, auto& b2, auto& A, auto& Z, auto& X_FFN, auto& w_output, auto& AT, vector<vector<float>> dh, auto& X_ATTN, auto& w_query, auto& w_key, auto& w_value, auto& query, auto& key, auto& value, auto& score, int embed_size, int head_size, float rate)
	{
	    int layers = w2.size();
	    int seq_len = dh.size();
	    int head_dim = embed_size / head_size;
	    float scale = 1.0f / sqrt(static_cast<float>(head_dim));

	    for (int i = layers - 1; i >= 0; --i)
	    {
	        auto dh_ffn_residual = dh;

	        auto A_T = Tensor::transpose(A[i]);
	        auto dw2 = Tensor::dot_product(A_T, dh);
	        auto db2 = Tensor::bias(dh);

	        auto w2_T = Tensor::transpose(w2[i]);
	        auto d_activation = Tensor::dot_product(dh, w2_T);
	        
	        Functions::gelu_derivative(Z[i]);
	        auto dz_ffn = d_activation;
	        for (size_t t = 0; t < seq_len; ++t) for (size_t j = 0; j < Z[i][0].size(); ++j) dz_ffn[t][j] *= Z[i][t][j];

	        auto X_FFN_T = Tensor::transpose(X_FFN[i]);
	        auto dw1 = Tensor::dot_product(X_FFN_T, dz_ffn);
	        auto db1 = Tensor::bias(dz_ffn);

	        auto w1_T = Tensor::transpose(w1[i]);
	        auto dX_ffn_path = Tensor::dot_product(dz_ffn, w1_T);

	        Functions::update(w2[i], dw2, rate);
	        Functions::update(w1[i], dw1, rate);
	        Functions::update(b2[i], db2, rate);
	        Functions::update(b1[i], db1, rate);

	        dh = Tensor::sum(dX_ffn_path, dh_ffn_residual);

	        auto dh_attn_residual = dh;

	        auto AT_T = Tensor::transpose(AT[i]);
	        auto dw_output = Tensor::dot_product(AT_T, dh);
	        auto w_output_t = Tensor::transpose(w_output[i]);
	        auto da_concate = Tensor::dot_product(dh, w_output_t);
	        auto da_concate_m = Tensor::reshape_to_multihead(da_concate, head_size);
			auto X_ATTN_T = Tensor::transpose(X_ATTN[i]);
	        
	        vector<vector<vector<float>>> dwq(head_size), dwk(head_size), dwv(head_size);
	        vector<vector<float>> dX_attn_path(seq_len, vector<float>(embed_size, 0.0f));

	        for (int j = 0; j < head_size; ++j)
	        {
	            auto score_t = Tensor::transpose(score[i][j]);
	            auto dv_head = Tensor::dot_product(score_t, da_concate_m[j]);

	            auto value_t = Tensor::transpose(value[i][j]);
	            auto dscore_h = Tensor::dot_product(da_concate_m[j], value_t);
	            auto dz_attn = dscore_h; 
	            for (int t = 0; t < seq_len; ++t)
	            {
	                float row_dot = 0.0f;
	                for (int k = 0; k < seq_len; ++k) row_dot += dscore_h[t][k] * score[i][j][t][k];
	                for (int k = 0; k < seq_len; ++k)
	                {
	                    dz_attn[t][k] = score[i][j][t][k] * (dscore_h[t][k] - row_dot);
	                    dz_attn[t][k] *= scale;
	                }
	            }

	            auto dq = Tensor::dot_product(dz_attn, key[i][j]);
	            auto dz_T = Tensor::transpose(dz_attn);
	            auto dk = Tensor::dot_product(dz_T, query[i][j]);

	            dwq[j] = Tensor::dot_product(X_ATTN_T, dq);
	            dwk[j] = Tensor::dot_product(X_ATTN_T, dk);
	            dwv[j] = Tensor::dot_product(X_ATTN_T, dv_head);
	            
	            auto wq_T = Tensor::transpose(w_query[i][j]);
	            auto wk_T = Tensor::transpose(w_key[i][j]);
	            auto wv_T = Tensor::transpose(w_value[i][j]);

	            auto dq_p = Tensor::dot_product(dq, wq_T);
	            auto dk_p = Tensor::dot_product(dk, wk_T);
	            auto dv_p = Tensor::dot_product(dv_head, wv_T);

				for(int t=0; t<seq_len; ++t) {
				    for(int d=0; d<embed_size; ++d) {
				        dX_attn_path[t][d] += dq_p[t][d] + dk_p[t][d] + dv_p[t][d];
				    }
				}
	        }

	        Functions::update(w_output[i], dw_output, rate);
	        Functions::update(w_query[i], dwq, rate);
	        Functions::update(w_key[i], dwk, rate);
	        Functions::update(w_value[i], dwv, rate);

	        dh = Tensor::sum(dX_attn_path, dh_attn_residual);
	    }
	    return dh;
	}
};
#endif