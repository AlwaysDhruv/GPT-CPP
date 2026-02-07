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
				for (size_t j = 0; j < vector1[0].size(); ++j)
				{
					sum += vector1[j][i] * vector2[k][j];
				}
				temp.push_back(sum);
			}
			vector3.push_back(temp);
		}
		return vector3;
	}
};
#endif