#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <fstream>
#include "Display.hpp"
#include "Tensor.cuh"
#include "../utils/ini.h"
#include <cuda_runtime.h>

class Transformer
{
	int embed_size;
	int vocab_size;
	int seq_lengh;
	int batch_size;
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
			seq_lengh = ids.size();
			token_ids = ids;
			cout << "Parameters imported from config.ini...." << endl;
		}
		else cout << "File Have Problem.." << endl;
	}

	void ready()
	{
		auto embed_matirx = GPU::random(embed_size * vocab_size);
		auto position_matirx = GPU::random(seq_lengh * embed_size);
		
		vector<float> context;
		context.reserve(seq_lengh * embed_size);

		for (int i = 0; i < seq_lengh; ++i)
		{
			int temp_index = (token_ids[i]) * embed_size;
			for (int j = temp_index; j < temp_index + embed_size; ++j) context.push_back(embed_matirx[j]);
		}
		auto X = GPU::add(context, position_matirx);
	}
};

#endif