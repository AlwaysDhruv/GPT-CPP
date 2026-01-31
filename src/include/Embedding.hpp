#ifndef VALUES_H
#define VALUES_H

#include <iostream>
#include <random>
#include <vector>
#include <cmath>

using namespace std;

class Embedd
{
public:
	void embeddings(vector<vector<float>>& values, int vocab_size, int dim_size)
	{
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<float> dist(-1.0, 1.0);

		for (size_t i = 0; i < vocab_size; ++i)
		{
			vector<float> temp;
			for (size_t j = 0; j < dim_size; ++j) temp.push_back(round(dist(gen) * 100) / 100);
			values.push_back(temp);
		}
	}

	void positioning_encoding(vector<vector<float>>& embedding)
	{
		int dim = embedding[0].size();
		
		for (size_t i = 0; i < embedding.size(); ++i)
		{
			float values;
			vector<float> temp;
			for (size_t j = 0; j < embedding[i].size(); ++j)
			{
				if (j % 2 == 0)
				{
					values = embedding[i][j] + sin(i / pow(10000.0, j / (float)dim));
					temp.push_back(round(values * 100) / 100);
				}
				else
				{
					values = embedding[i][j] + cos(i / pow(10000.0, (j - 1) / (float)dim));
					temp.push_back(round(values * 100) / 100);
				}
			}
			embedding[i] = temp;
		}
	}
};

#endif