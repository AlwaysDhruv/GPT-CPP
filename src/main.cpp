#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
#include <cmath>

#include "./include/BPE.hpp"
#include "./include/Embedding.hpp"
#include "./include/Tensor.hpp"
#include "./include/Display.hpp"
#include "./include/Functions.hpp"
#include "./include/Attension.hpp"
#include "./include/ForwardPass.hpp"
#include "./include/BackwardPass.hpp"
#include "./utils/ini.h"

using namespace std;

int main(int argc, char const *argv[])
{
	mINI::INIFile file("../config.ini");
    mINI::INIStructure ini;

	if(file.read(ini))
	{
		int embed_size = stoi(ini["GPT"]["Emdedding_size"]);
		int head_size = stoi(ini["GPT"]["Head_size"]);\
		int layers = stoi(ini["GPT"]["Layers"]);
		int vocab_size = stoi(ini["GPT"]["Vocab_size"]);
		float learning_rate = stof(ini["GPT"]["Rate"]);
		int embed_size2 = embed_size * embed_size;

		Tokenize tk;
		vector<string> tokens;
		vector<long long> token_ids;
		tk.encoding("../data/test2.txt", tokens, token_ids);
		
		int seq_len = token_ids.size();

		vector<long long> X(seq_len - 1);
		vector<long long> Y(seq_len - 1);
		for (size_t i = 0; i < seq_len - 1; ++i) X[i] = token_ids[i];
		for (size_t i = 1; i < seq_len; ++i) Y[i - 1] = token_ids[i];

		Embedd ed;
		vector<vector<float>> X_IN(seq_len - 1, vector<float>(embed_size, 0.0f));
		ed.embeddings(X_IN);	
		ed.positioning_encoding(X_IN);

		auto X_INT = X_IN;

		auto w_query = Functions::linear(layers ,embed_size);
		auto w_key = Functions::linear(layers, embed_size);
		auto w_value = Functions::linear(layers, embed_size);
		auto w_output = Functions::linear(layers, embed_size);
		auto w_lm = Functions::linear2(embed_size, vocab_size);
		auto w1 = Functions::linear(layers, embed_size, embed_size2);
		auto w2 = Functions::linear(layers, embed_size2, embed_size);
		
		vector<vector<float>>b1(layers, vector<float>(embed_size2, 0.0f));
		vector<vector<float>>b2(layers, vector<float>(embed_size, 0.0f));
		vector<float>b_lm(vocab_size, 0.0f);

		vector<vector<vector<float>>> A(layers, vector<vector<float>>(seq_len,vector<float>(embed_size2, 0.0f)));
		vector<vector<vector<float>>> Z(layers, vector<vector<float>>(seq_len,vector<float>(embed_size2, 0.0f)));
		vector<vector<float>> H(seq_len, vector<float>(embed_size));
		vector<vector<vector<float>>> X_IN2(layers, vector<vector<float>>(seq_len, vector<float>(embed_size)));

		vector<vector<vector<float>>> AT(layers, vector<vector<float>>(seq_len,vector<float>(embed_size, 0.0f)));
		vector<vector<vector<vector<float>>>> value(layers, vector<vector<vector<float>>>(head_size ,vector<vector<float>>(seq_len,vector<float>(embed_size / head_size, 0.0f))));
		vector<vector<vector<vector<float>>>> score(layers, vector<vector<vector<float>>>(head_size, vector<vector<float>>(seq_len,vector<float>(seq_len, 0.0f))));
		vector<vector<vector<vector<float>>>> da_h(layers, vector<vector<vector<float>>>(head_size ,vector<vector<float>>(seq_len,vector<float>(embed_size / head_size, 0.0f))));

		Forward::Layers(ini, X_INT, A, H, Z, AT, score, w_query, w_key, w_value, w_output, w1, w2, w_lm, b1, b2, b_lm, value, X_IN2);
		
		auto dz = Functions::gradient_loss(X_INT, Y);

		auto dh = Backward::backward_LM(w_lm, b_lm, H, dz, learning_rate);

		dh = Backward::backward_FFN(w1, w2, b1, b2, A, Z, AT, X_IN2, dh, learning_rate);
		
		auto dv_h = Backward::backward_Attension1(w_output, AT, dh, da_h, score, head_size, learning_rate);
		
		Backward::backward_Attension2(da_h, value, score);
	}
	else cout << "File Have Problem.." << endl;

	return 0;
}