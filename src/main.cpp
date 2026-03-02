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
		int head_size = stoi(ini["GPT"]["Head_size"]);
		int layers = stoi(ini["GPT"]["Layers"]);
		int vocab_size = stoi(ini["GPT"]["Vocab_size"]);

		Tokenize tk;
		vector<string> tokens;
		vector<long long> token_ids;
		tk.encoding("../data/test2.txt", tokens, token_ids);

		vector<long long> X(token_ids.size() - 1);
		vector<long long> Y(token_ids.size() - 1);
		for (size_t i = 0; i < token_ids.size() - 1; ++i) X[i] = token_ids[i];
		for (size_t i = 1; i < token_ids.size(); ++i) Y[i - 1] = token_ids[i];

		Embedd ed;
		vector<vector<float>> X_IN(X.size(), vector<float>(embed_size, 0.0f));
		ed.embeddings(X_IN);
		ed.positioning_encoding(X_IN);

		auto w_query = Functions::linear(layers ,embed_size);
		auto w_key = Functions::linear(layers, embed_size);
		auto w_value = Functions::linear(layers, embed_size);
		auto w_output = Functions::linear(layers, embed_size);
		auto w_lm = Functions::linear2(embed_size, vocab_size);
		auto w1 = Functions::linear(layers, embed_size, embed_size * embed_size);
		auto w2 = Functions::linear(layers, embed_size * embed_size, embed_size);
		
		vector<vector<float>>b1(layers, vector<float>(embed_size * embed_size, 0.0f));
		vector<vector<float>>b2(layers, vector<float>(embed_size, 0.0f));
		vector<float>b_lm(vocab_size, 0.0f);
		

		auto H = Forward::Layers(X_IN, w_query, w_key, w_value, w_output, w1, w2, w_lm, b1, b2, b_lm, head_size, vocab_size, layers);
		
		auto dz = Functions::gradient_loss(X_IN, Y);
		
		Backward::backward(w_lm, dz, H);
	}
	else cout << "File Have Problem.." << endl;

	return 0;
}