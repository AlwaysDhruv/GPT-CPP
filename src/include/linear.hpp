#ifndef LINEAR_PROJECTION
#define LINEAR_PROJECTION

#include <iostream>
#include <random>
#include <vector>
#include <cmath>

using namespace std;

class Linear
{
public:
	void weigths(vector<vector<float>>& weigths, int dim_size)
	{
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<float> dist(-1.0, 1.0);

		for (size_t i = 0; i < dim_size; ++i)
		{
			vector<float> temp;
			for (size_t j = 0; j < dim_size; ++j) temp.push_back(round(dist(gen) * 100) / 100);
			weigths.push_back(temp);
		}
	}
};

#endif