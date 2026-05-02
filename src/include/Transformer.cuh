#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <fstream>
#include "Tensor.cuh"
#include "Display.hpp"
#include "../utils/ini.h"

class Transformer
{
	int embed_size;
	int vocab_size;
	int seq_len;
	int batch_size;
	int context_len;
	int num_seq;
	int xy_size;
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
			xy_size = context_len - 1;
			num_seq = xy_size - seq_len;
			cout << "Parameters imported from config.ini...." << endl;
		}
		else cout << "config.ini Have Problem...." << endl;
	}

	void ready()
	{
		vector<long long> token_x;
		vector<long long> token_y;
		
		token_x.reserve(xy_size);
		token_y.reserve(xy_size);

		for (int i = 0, j = 1; i < xy_size, j < context_len; ++i, ++j)
		{
			token_x.push_back(token_ids[i]);
			token_y.push_back(token_ids[j]);	
		}

		auto embed_mat = GPU::random(vocab_size * embed_size);
		auto pos_mat = GPU::random(xy_size * embed_size);

		vector<float> input;
		input.reserve(xy_size * embed_size);
		
		for (int i = 0; i < xy_size; ++i)
		{
			int temp_index = token_x[i] * embed_size;
			for (int j = temp_index; j < temp_index + embed_size; ++j) input.push_back(embed_mat[j]);
		}
    auto X = GPU::add(input, pos_mat);

		cout << "Batching....." << endl;
		int ct = 0;
		for (int i = 0; i < num_seq; i+=batch_size)
		{
			cout << "Batch " << ++ct << endl;
			cout << "==================================" << endl;
			for (int batch = 0; batch < batch_size && (i + batch) < num_seq ; ++batch)
			{
				int temp_index = i + batch;

			    for (int j = 0; j < seq_len; ++j)
			    {
			        int base = (temp_index + j) * embed_size;

			        for (int k = 0; k < embed_size; ++k)
			        {
			            cout << X[base + k] << " ";
			        }
			        cout << endl;
			    }
			    cout << endl;
			}
			cout << "==================================" << endl;
			cout << endl << endl;
		}
	}
};

#endif