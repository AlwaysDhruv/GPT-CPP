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
#include "./include/Transformer.hpp"
#include "./utils/ini.h"

using namespace std;

int main(int argc, char const *argv[])
{

	Tokenize tk;
	vector<string> tokens;
	vector<long long> token_ids;
	tk.encoding("../data/test2.txt", tokens, token_ids);
	//cout << token_ids.size() << endl;
	Transformer tr(token_ids);
	tr.fit(50);	
	return 0;
}