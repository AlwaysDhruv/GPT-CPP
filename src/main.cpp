#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>

#include "./include/BPE.hpp"
#include "./include/Embedding.hpp"
#include "./include/Linear.hpp"
#include "./include/Tensor.hpp"
using namespace std;

int main(int argc, char const *argv[])
{
	Tokenize tk;
	vector<string> tokens;
	vector<long long> token_ids;
	tk.encoding("../data/test2.txt", tokens, token_ids);
	
	Embedd ed;
	vector<vector<float>> embedding;
	ed.embeddings(embedding, token_ids.size(), 4);
	ed.positioning_encoding(embedding);

	Linear projection;
	vector<vector<float>> w_query;
	vector<vector<float>> w_key;
	vector<vector<float>> w_value;
	projection.linear(w_query, w_key, w_value, 4);
	
	Tensor tensr;
	w_query = tensr.dot_product(w_query, embedding);
	w_key = tensr.dot_product(w_key, embedding);
	w_value = tensr.dot_product(w_value, embedding);

	vector<vector<vector<float>>> query;
	vector<vector<vector<float>>> key;
	vector<vector<vector<float>>> value;
	query = tensr.multi_head(w_query, 2);
	key = tensr.multi_head(w_key, 2);
	value = tensr.multi_head(w_value, 2);
	return 0;
}