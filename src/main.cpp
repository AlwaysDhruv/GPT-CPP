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
#include "./include/Layers.hpp"
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

		Tokenize tk;
		vector<string> tokens;
		vector<long long> token_ids;
		tk.encoding("../data/test.txt", tokens, token_ids);

		Embedd ed;
		vector<vector<float>> X_IN(token_ids.size(), vector<float>(embed_size, 0.0f));
		ed.embeddings(X_IN);
		ed.positioning_encoding(X_IN);

		auto X_NORM = Layer::normalizaion(X_IN);	
		
		auto w_query = Layer::linear(embed_size);
		auto w_key = Layer::linear(embed_size);
		auto w_value = Layer::linear(embed_size);
		auto w_output = Layer::linear(embed_size);

		Tensor tensr;
		w_query = tensr.projection(w_query, X_IN);
		w_key = tensr.projection(w_key, X_IN);
		w_value = tensr.projection(w_value, X_IN);

		auto query = tensr.reshape_to_multihead(w_query, head_size);
		auto key = tensr.reshape_to_multihead(w_key, head_size);
		auto value = tensr.reshape_to_multihead(w_value, head_size);

		Multiheadattension block;
		auto Z = block.decoder(query, key, value);

		Z = tensr.dot_product(Z, w_output);

		auto X_NEW = tensr.sum(Z, X_IN);
		Debug::shape(X_NEW);
	}

	else cout << "File Have Problem.." << endl;

	return 0;
}