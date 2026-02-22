#ifndef TENSOR
#define TENSOR

class Tensor
{
public:
	vector<vector<float>> dot_product(vector<vector<float>> vector1, vector<vector<float>> vector2)
	{
		vector<vector<float>> vector3;
		for (size_t k = 0; k < vector2.size(); ++k)
		{
			vector<float> temp;
			for (size_t i = 0; i < vector1.size(); ++i)
			{
				float sum = 0.0;
				for (size_t j = 0; j < vector1[0].size(); ++j) sum += vector1[j][i] * vector2[k][j];
				temp.push_back(sum);
			}
			vector3.push_back(temp);
		}
		return vector3;
	}
	
	vector<vector<vector<float>>> multi_head(vector<vector<float>> value, int head_size)
	{
		int dim_size = value[0].size() / head_size;
		vector<vector<vector<float>>> values;
		
		for (size_t i = 0; i < value.size(); ++i)
		{
			int ct = 0;
			vector<float> temp;
			vector<vector<float>> temp1;
			for (size_t j = 0; j < value[0].size(); ++j)
			{
				if (ct==dim_size)
				{
					temp1.push_back(temp);
					temp.clear();
					ct = 0;
				}
				temp.push_back(value[i][j]);
				++ct;
			}
			for (size_t j = value[0].size() - dim_size; j < value[0].size(); ++j) temp.push_back(value[i][j]);
			temp1.push_back(temp);
			values.push_back(temp1);
		}
		return values;
	}
};
#endif