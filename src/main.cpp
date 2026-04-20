#include <iostream>
#include <string>
#include <vector>
#include "./include/BPE.hpp"
#include "./include/Transformer.cu"
using namespace std;

int main(int argc, char const *argv[])
{
	Tokenization tk;
	vector<string> tokens;
	vector<long long> token_ids;
	//tk.fit("../data/test.txt", 72);
	tk.encoding("../data/test.txt", tokens, token_ids);
	Transformer tr(token_ids);
	tr.ready();
	return 0;
}