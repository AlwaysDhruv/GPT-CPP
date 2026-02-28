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
	void embeddings(vector<vector<float>>& values)
	{
		random_device rd;
		mt19937 gen(rd());
		uniform_real_distribution<float> dist(-1.0, 1.0);

		for (size_t i = 0; i < values.size(); ++i) for (size_t j = 0; j < values[0].size(); ++j) values[i][j] = dist(gen);
	}

	void positioning_encoding(vector<vector<float>>& embedding)
	{
		int dim = embedding[0].size();
		
		for (size_t i = 0; i < embedding.size(); ++i)
		{
			float values;
			for (size_t j = 0; j < embedding[i].size(); ++j)
			{
				if (j % 2 == 0) embedding[i][j] = embedding[i][j] + sin(i / pow(10000.0, j / (float)dim));
				else embedding[i][j] = embedding[i][j] + cos(i / pow(10000.0, (j - 1) / (float)dim));
			}
		}
	}
};

#endif