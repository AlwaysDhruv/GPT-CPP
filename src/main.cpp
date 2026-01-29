#include <iostream>
#include <random>
#include <vector>
#include <cmath>

#include "./include/BPE.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
	Tokenize tk;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<float> dist(-1.0, 1.0);
	
	cout << round(dist(gen) * 100) / 100 << endl;
	
	return 0;
}