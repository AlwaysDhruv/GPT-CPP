#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <cmath>

#include "./include/BPE.hpp"
#include "./include/Embedding.hpp"
#include "./include/Linear.hpp"

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
	vector<vector<float>> weigths;
	projection.weigths(weigths, 12);

	cout << weigths.size() << " " << weigths[0].size() << endl;
	
	return 0;
}