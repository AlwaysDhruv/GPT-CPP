#ifndef ATTENSION
#define ATTENSION

#include <iostream>
#include <vector>
#include <cmath>
#include "Tensor.hpp"
#include "Display.hpp"
using namespace std;

class Attension
{
public:
	void scores(vector<vector<vector<float>>> query, vector<vector<vector<float>>> key, vector<vector<vector<float>>> value)
	{
		Debug::display(query[0]);
		Tensor tnsr;
		Debug::display(key[0]);
		for (size_t j = 0; j < query.size(); ++j)
		{
			key[j] = tnsr.transpose(key[j]);
			for (size_t i = 0; i < query[0].size(); ++i) // 6
			{
				for (size_t k = 0; k < key[0][0].size(); ++k) //6
				{
					float sum = 0.0f;
					for (size_t l = 0; l < key[0].size(); ++l)
					{
						sum += query[j][i][l] * key[j][l][k];
					}
					cout << sum << " ";
				}
				cout << endl;
			}
			cout << endl;
		}
	}
};
#endif