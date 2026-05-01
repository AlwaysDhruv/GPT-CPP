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

		int xy_size = context_len - 1;
		
		token_x.reserve(xy_size);
		token_y.reserve(xy_size);

		for (int i = 0, j = 1; i < xy_size, j < context_len; ++i, ++j)
		{
			token_x.push_back(token_ids[i]);
			token_y.push_back(token_ids[j]);	
		}

		auto embed_mat = Initial::weights(vocab_size * embed_size);
		auto pos_mat = Initial::weights((xy_size) * embed_size);

		vector<float> input;
		input.reserve((xy_size) * embed_size);
		
		for (int i = 0; i < xy_size; ++i)
		{
			int temp_index = token_x[i] * embed_size;
			for (int j = temp_index; j < temp_index + embed_size; ++j) input.push_back(embed_mat[j]);
		}

		for (int i = 0; i < xy_size; ++i)
		{
			int temp_index = i * embed_size;
			for (int k = temp_index; k < temp_index + embed_size; ++k)
			{
				cout << input[k] << " ";
			}
			cout << endl;
		}
		
		cout << endl << "Sequencing....." << endl;

		int ct = 0;
		for (int i = 0; i < xy_size - 1 && i * (seq_len - 1) < xy_size - 1; ++i)
		{
			int temp_index_s = i * (seq_len - 1);

			for (int j = temp_index_s; j < temp_index_s + seq_len; ++j)
			{
				int temp_index = j * embed_size;
				for (int k = temp_index; k < temp_index + embed_size; ++k)
				{
					cout << input[k] << " ";
				}
				cout << endl;
			}
			cout << endl << endl;
			++ct;
		}
		cout << ct << endl
	}
};

#endif