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
#include "./utils/ini.h"

using namespace std;

int main(int argc, char const *argv[])
{
	mINI::INIFile file("../config.ini");
    mINI::INIStructure ini;

	if(file.read(ini))
	{
		int embed_size = stoi(ini["GPT"]["Emdedding_size"]);
		int head_size = stoi(ini["GPT"]["Head_size"]);
		int layers = stoi(ini["GPT"]["Layers"]);

		Tokenize tk;
		vector<string> tokens;
		vector<long long> token_ids;
		tk.encoding("../data/test2.txt", tokens, token_ids);

		Embedd ed;
		vector<vector<float>> X_IN(token_ids.size(), vector<float>(embed_size, 0.0f));
		ed.embeddings(X_IN);
		ed.positioning_encoding(X_IN);

		auto w_query = Functions::linear(layers ,embed_size);
		auto w_key = Functions::linear(layers, embed_size);
		auto w_value = Functions::linear(layers, embed_size);
		auto w_output = Functions::linear(layers, embed_size);
		auto w1 = Functions::linear(layers, embed_size, embed_size * embed_size);
		auto w2 = Functions::linear(layers, embed_size * embed_size, embed_size);
		vector<vector<float>>b1(layers, vector<float>(embed_size * embed_size));
		vector<vector<float>>b2(layers, vector<float>(embed_size));
		
		for (size_t i = 0; i < layers; ++i)
		{
			auto X_NORM = Functions::normalizaion(X_IN);

			Tensor tensr;
			w_query[i] = tensr.projection(w_query[i], X_NORM);
			w_key[i] = tensr.projection(w_key[i], X_NORM);
			w_value[i] = tensr.projection(w_value[i], X_NORM);
			
			auto query = tensr.reshape_to_multihead(w_query[i], head_size);
			auto key = tensr.reshape_to_multihead(w_key[i], head_size);
			auto value = tensr.reshape_to_multihead(w_value[i], head_size);


			//Attension
			Multiheadattension block;
			
			auto Z = block.score(query, key, value);
			Z = tensr.dot_product(Z, w_output[i]);

			auto X_NEW = tensr.sum(Z, X_IN);

			//FFN
			X_NORM = Functions::normalizaion(X_NEW);

			auto H = tensr.dot_product(X_NORM, w1[i]);
			tensr.sum(H, b1[i]);

			Functions::gelu(H);
		
			auto Z_FFN = tensr.dot_product(H, w2[i]);
			tensr.sum(Z_FFN, b2[i]);

			X_IN = tensr.sum(Z_FFN, X_NEW);
		}

		Debug::display(X_IN);
	}
	else cout << "File Have Problem.." << endl;

	return 0;
}