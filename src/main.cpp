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
	tk.encoding("../data/test.txt", tokens, token_ids);
	
	Embedd ed;
	
	vector<vector<float>> embedding;
	ed.embeddings(embedding, token_ids.size(), 12);
	ed.positioning_encoding(embedding);

	Linear projection;
	vector<vector<float>> w_query;
	vector<vector<float>> w_key;
	vector<vector<float>> w_value;
	projection.linear(w_query, w_key, w_value, 12);
	
	Tensor tensr;
	vector<vector<float>> query;
	vector<vector<float>> key;
	vector<vector<float>> value;
	query = tensr.dot_product(w_query, embedding);
	key = tensr.dot_product(w_key, embedding);
	value = tensr.dot_product(w_value, embedding);
	return 0;
}