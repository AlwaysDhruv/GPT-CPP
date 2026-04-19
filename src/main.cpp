#include <iostream>
#include <string>
#include <vector>
#include "./include/BPE.hpp"
#include "./include/Transformer.hpp"
using namespace std;

int main(int argc, char const *argv[])
{
	Tokenization tk;
	vector<string> tokens;
	vector<long long> token_ids;
	//tk.fit("../data/test.txt", 72);
	tk.encoding("../data/test.txt", tokens, token_ids);
	Transformer tr;
	return 0;
}