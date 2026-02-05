#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>

#include "./include/BPE.hpp"
#include "./include/Embedding.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
	Tokenize tk;
	
	vector<string> tokens;
	vector<long long> token_ids;
	tk.encoding("../data/test.txt", tokens, token_ids);
	
	Embedd ed;
	
	vector<vector<float>> embedding;
	ed.embeddings(embedding, token_ids.size(), 10);
	ed.positioning_encoding(embedding);

	
	return 0;
}