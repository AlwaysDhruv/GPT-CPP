#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "../utils/ini.h"
#include <iostream>
#include <fstream>
#include "../utils/json.hpp"

using json = nlohmann::json;

class Transformer
{
	int embed_size;
	int head_size;
	int layers;
	int vocab_size;
	float learning_rate;
	int embed_size2;
	int head_dim;
	int seq_len;
	int context_len;
	vector<long long> token_ids;
	vector<long long> X;
	vector<long long> Y;
	mINI::INIStructure ini;
public:
	
	Transformer(auto token_id)
	{
		mINI::INIFile file("../config.ini");
	    mINI::INIStructure in;

		if(file.read(in))
		{
			embed_size = stoi(in["GPT"]["Emdedding_size"]);
			head_size = stoi(in["GPT"]["Head_size"]);
			layers = stoi(in["GPT"]["Layers"]);
			vocab_size = stoi(in["GPT"]["Vocab_size"]);
			learning_rate = stof(in["GPT"]["Rate"]);
			embed_size2 = embed_size * embed_size;
			head_dim = embed_size / head_size;
			context_len = token_id.size();
			seq_len = context_len - 1;
			token_ids = token_id;
			ini = in;
			X.assign(seq_len, 0);
			Y.assign(seq_len, 0);
			for (size_t i = 0; i < seq_len; ++i) X[i] = token_ids[i];
			for (size_t i = 1; i < context_len; ++i) Y[i - 1] = token_ids[i];
		}
		else cout << "File Have Problem.." << endl;
	}

	void fit(int epochs)
	{		
		Embedd ed;		

		auto embbed_matrix = ed.init_weights(vocab_size, embed_size);
		auto pos_matrix = ed.init_weights(seq_len, embed_size);

		auto w_query = Functions::linear3(layers , head_size, embed_size, head_dim);
		auto w_key = Functions::linear3(layers , head_size, embed_size, head_dim);;
		auto w_value = Functions::linear3(layers , head_size, embed_size, head_dim);;
		auto w_output = Functions::linear(layers, embed_size);
		auto w1 = Functions::linear(layers, embed_size, embed_size2);
		auto w2 = Functions::linear(layers, embed_size2, embed_size);
		vector<vector<float>>b1(layers, vector<float>(embed_size2, 0.0f));
		vector<vector<float>>b2(layers, vector<float>(embed_size, 0.0f));
		vector<float>b_lm(vocab_size, 0.0f);
		vector<vector<float>> gamma(layers, vector<float>(embed_size, 1.0f));
		vector<vector<float>> beta(layers, vector<float>(embed_size, 0.0f));		
		
		for (int epoch = 0; epoch < epochs; ++epoch)
		{
			auto X_IN = ed.forward(X, embbed_matrix);	
			ed.positioning_encoding(X_IN, pos_matrix);

			vector<vector<vector<float>>> A(layers, vector<vector<float>>(seq_len,vector<float>(embed_size2, 0.0f)));
			vector<vector<vector<float>>> Z(layers, vector<vector<float>>(seq_len,vector<float>(embed_size2, 0.0f)));
			vector<vector<float>> H(seq_len, vector<float>(embed_size));
			vector<vector<vector<float>>> X_IN2(layers, vector<vector<float>>(seq_len, vector<float>(embed_size)));
			vector<vector<vector<float>>> X_IN3(layers, vector<vector<float>>(seq_len, vector<float>(embed_size)));
			vector<vector<vector<float>>> AT(layers, vector<vector<float>>(seq_len,vector<float>(embed_size, 0.0f)));
			vector<vector<vector<vector<float>>>> query(layers, vector<vector<vector<float>>>(head_size ,vector<vector<float>>(seq_len,vector<float>(embed_size / head_size, 0.0f))));
			vector<vector<vector<vector<float>>>> key(layers, vector<vector<vector<float>>>(head_size ,vector<vector<float>>(seq_len,vector<float>(embed_size / head_size, 0.0f))));
			vector<vector<vector<vector<float>>>> value(layers, vector<vector<vector<float>>>(head_size ,vector<vector<float>>(seq_len,vector<float>(embed_size / head_size, 0.0f))));
			vector<vector<vector<vector<float>>>> score(layers, vector<vector<vector<float>>>(head_size, vector<vector<float>>(seq_len,vector<float>(seq_len, 0.0f))));
			vector<vector<float>> gradients(vocab_size, vector<float>(embed_size, 0.0f));

			auto P = Forward::Layers(ini, X_IN, A, H, Z, AT, score, w_query, w_key, w_value, w_output, w1, w2, embbed_matrix, b1, b2, b_lm, query, key, value, X_IN2, X_IN3);
					
			auto loss = Functions::loss(P, Y);

			cout << "Loss In " << epoch + 1 << " Step :- " << loss << endl;
			
			auto dz = Functions::gradient(P, Y);
			
			auto dz_t = Tensor::transpose(dz);

			gradients = Tensor::dot_product(dz_t, H);
			
			auto dh = Tensor::dot_product(dz, embbed_matrix);

			auto db_lm = Tensor::bias(dz);
			
			auto grad_pos = dh;

			dh = Backward::backward_Transformer(w1, w2, b1, b2, A, Z, X_IN2, w_output, AT, dh, X_IN3, w_query, w_key, w_value, query, key, value, score, embed_size, head_size, learning_rate);

			Functions::update(embbed_matrix, gradients, learning_rate);
			Functions::update(pos_matrix, grad_pos, learning_rate);
			Functions::update(b_lm, db_lm, learning_rate);
		}
	}
};
#endif