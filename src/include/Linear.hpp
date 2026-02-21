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
	void linear(vector<vector<float>>& query, vector<vector<float>>& key, vector<vector<float>>& value, int dim_size)
	{
		vector<vector<vector<float>>> weigths;
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<float> dist(-1.0, 1.0);
		for (size_t i = 0; i < 3; ++i)
		{
			vector<vector<float>> temp1;
			for (size_t i = 0; i < dim_size; ++i)
			{
				vector<float> temp2;
				for (size_t j = 0; j < dim_size; ++j) temp2.push_back(round(dist(gen) * 100) / 100);
				temp1.push_back(temp2);
			}
			weigths.push_back(temp1);
		}
		query = weigths[0];
		key = weigths[1];
		value = weigths[2];
	}
};

#endif