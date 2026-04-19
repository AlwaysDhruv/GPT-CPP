#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <fstream>
#include "../utils/ini.h"
#include "Initilization.hpp"

class Transformer
{
	int embed_size;
	int vocab_size;
	int seq_lengh;

public:
	
	Transformer(int sequence)
	{
		mINI::INIFile file("../config.ini");
	    mINI::INIStructure in;

		if(file.read(in))
		{
			embed_size = stoi(in["GPT"]["Emdedding_size"]);
			vocab_size = stoi(in["GPT"]["Vocab_size"]);
			seq_lengh = sequence;
			cout << "Parameters imported from config.ini...." << endl;
		}
		else cout << "File Have Problem.." << endl;
	}
	void ready()
	{
		auto embed_matirx = Initial::weights(embed_size * vocab_size);
		auto position_matirx = Initial::weights(seq_lengh * embed_size);
		
		cout << embed_matirx.size() << endl;
		cout << position_matirx.size() << endl;
	}
};

#endif