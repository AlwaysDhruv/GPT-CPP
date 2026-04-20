#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <iostream>
#include <fstream>
#include "../utils/ini.h"
#include "Initilization.hpp"
#include "Display.hpp"
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
		auto embed_matirx = Initial::weights(embed_size * vocab_size);
		auto position_matirx = Initial::weights(seq_lengh * embed_size);
		
		vector<vector<float>> context;
		context.reserve(seq_lengh);

		for (int i = 0; i < seq_lengh; ++i)
		{
			int temp_index = (token_ids[i] - 1) * embed_size;
			vector<float> v;
			v.reserve(embed_size);
			for (int j = temp_index, k = 0; j < temp_index + embed_size, k < embed_size; ++j, ++k) v.push_back(embed_matirx[j]);
			context.push_back(v);
		}
		Debug::shape(context);
		Debug::shape(position_matirx);
		
		int deviceCount = 0;
    	cudaError_t error = cudaGetDeviceCount(&deviceCount);

	    if (error != cudaSuccess) {
	        std::cout << "CUDA error: " << cudaGetErrorString(error) << std::endl;
	    }

	    if (deviceCount == 0) {
	        std::cout << "No NVIDIA GPUs found." << std::endl;
	    } else {
	        std::cout << "Found " << deviceCount << " GPU(s)." << std::endl;
	        for (int i = 0; i < deviceCount; ++i) {
	            cudaDeviceProp prop;
	            cudaGetDeviceProperties(&prop, i);
	            std::cout << "Device " << i << ": " << prop.name << std::endl;
	        }
	    }
	}
};

#endif