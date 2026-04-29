#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <fstream>
#include "Tensor.hpp"
#include "Display.hpp"
#include "../utils/ini.h"
#include "Initilization.hpp"

class Transformer
{
	int embed_size;
	int vocab_size;
	int seq_len;
	int batch_size;
	int context_len;
	vector<long long> token_ids;

public:
	
	Transformer(vector<long long>& ids)
	{
		mINI::INIFile file("../config.ini");
	    mINI::INIStructure in;

		if(file.read(in))
		{
			embed_size = stoi(in["GPT"]["Emdedding_size"]);
			vocab_size = stoi(in["GPT"]["Vocab_size"]);
			batch_size = stoi(in["GPT"]["Batch_size"]);
			seq_len = stoi(in["GPT"]["Seq_len"]);
			token_ids = ids;
			context_len = token_ids.size();

			cout << "Parameters imported from config.ini...." << endl;
		}
		else cout << "File Have Problem.." << endl;
	}

	void ready()
	{
		vector<long long> token_x;
		vector<long long> token_y;

		token_x.reserve(context_len - 1);
		token_y.reserve(context_len - 1);

		for (int i = 0, j = 1; i < context_len - 1, j < context_len; ++i, ++j)
		{
			token_x.push_back(token_ids[i]);
			token_y.push_back(token_ids[j]);	
		}

		auto embed_mat = Initial::weights(vocab_size * embed_size);
		auto pos_mat = Initial::weights((context_len - 1) * embed_size);

		vector<float> input;
		input.reserve((context_len - 1) * embed_size);
		
		for (int i = 0; i < context_len - 1; ++i)
		{
			int temp_index = token_x[i] * embed_size;
			for (int j = temp_index; j < temp_index + embed_size; ++j) input.push_back(embed_mat[j]);
		}

		for (int i = 0; i < context_len - 1; ++i)
		{
			int temp_index = i * embed_size;

			for (int j = temp_index; j < temp_index + embed_size; ++j)
			{
				cout << input[j] << " ";
			}
			cout << endl;
		}
	}
};

#endif