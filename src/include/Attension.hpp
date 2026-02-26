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
		Tensor tensr;
		key[0] = tensr.transpose(key[0]);
		Debug::display(key[0]);
		
	}
};
#endif