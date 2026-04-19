#ifndef WEIGTHS_H
#define WEIGTHS_H

#include <iostream>
#include <random>
#include <vector>

using namespace std;

namespace Initial
{
	vector<float> weights(int n)
	{
		vector<float> weight(n, 0);
		random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<float> dist(0.0, 1.0);
        for (int i = 0; i < n; ++i) weight[i] = dist(gen);

        return weight;
	}
};

#endif