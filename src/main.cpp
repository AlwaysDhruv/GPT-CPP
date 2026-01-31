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

	for (size_t i = 0; i < embedding.size(); ++i)
	{
		for (size_t j = 0; j < embedding[i].size(); ++j)
		{
			cout << embedding[i][j] << " ";
		}
		cout << endl;
	}
	return 0;
}