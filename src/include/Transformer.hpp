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
		vector<vector<long long>> chunks;
		
		chunks.reserve(context_len - seq_len);

		for (int i = 0; i < context_len - seq_len; ++i)
		{
			vector<long long> temp;
			temp.reserve(seq_len + ct + 1);

			for (int j = ct; j < seq_len + ct + 1; ++j) temp.push_back(token_ids[j]);
			++ct;
			chunks.push_back(temp);
		}

		Debug::display(chunks);
		
		vector<vector<long long>> token_x;
		vector<vector<long long>> token_y;

		token_x.reserve(context_len - seq_len);
		token_y.reserve(context_len - seq_len);

		for (int k = 0; k < context_len - seq_len; ++k)
		{
			vector<long long> token_x_temp;
			vector<long long> token_y_temp;
			
			token_x_temp.reserve(seq_len);
			token_y_temp.reserve(seq_len);
			
			for (int i = 0, j = 1; i < chunks[k].size() - 1, j < chunks[k].size(); ++i, ++j)
			{
				token_x_temp.push_back(chunks[k][i]);
				token_y_temp.push_back(chunks[k][j]);
			}
			token_x.push_back(token_x_temp);
			token_y.push_back(token_y_temp);
		}
		Debug::display(token_x);
		Debug::display(token_y);
	}
};

#endif