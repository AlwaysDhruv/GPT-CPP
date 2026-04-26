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
		int ct = 0;
		int cols = context_len - seq_len;
		vector<vector<long long>> chunks;
		
		chunks.reserve(cols);

		for (int i = 0; i < cols; ++i)
		{
			vector<long long> temp;
			temp.reserve(seq_len + ct + 1);

			for (int j = ct; j < seq_len + ct + 1; ++j) temp.push_back(token_ids[j]);
			++ct;
			chunks.push_back(temp);
		}
		
		int rows = chunks[0].size() - 1;

		vector<long long> token_x;
		vector<long long> token_y;

		token_x.reserve(cols * rows);
		token_y.reserve(cols * rows);
		
		for (int k = 0; k < cols; ++k)
		{			
			for (int i = 0, j = 1; i < rows, j < rows + 1; ++i, ++j)
			{
				token_x.push_back(chunks[k][i]);
				token_y.push_back(chunks[k][j]);
			}
		}
	}
};

#endif