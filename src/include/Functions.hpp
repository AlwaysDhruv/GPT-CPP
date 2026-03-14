#ifndef FUNCTIONS
#define FUNCTIONS

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

namespace Functions
{
    void causal_mask(vector<vector<float>>& attention_score)
    {
        for (size_t i = 0; i < attention_score.size(); ++i)
            for (size_t j = 0; j < attention_score[i].size(); ++j)
                if (j > i) attention_score[i][j] = -1e9f;
    }

    void scaling(vector<vector<float>>& attention_score, float scale)
    {
        for (size_t i = 0; i < attention_score.size(); ++i)
            for (size_t j = 0; j < attention_score[i].size(); ++j)
                attention_score[i][j] *= scale;
    }

    void softmax(vector<vector<float>>& matrix)
    {
        int rows = matrix.size();
        if (rows == 0) return;

        int cols = matrix[0].size();

        for (int i = 0; i < rows; ++i)
        {
            float max_val = *max_element(matrix[i].begin(), matrix[i].end());

            float sum = 0.0f;

            for (int j = 0; j < cols; ++j)
            {
                matrix[i][j] = exp(matrix[i][j] - max_val);
                sum += matrix[i][j];
            }

            for (int j = 0; j < cols; ++j)
                matrix[i][j] /= sum + 1e-9f;
        }
    }
    
    float gelu_single(float x)
    {
        const float sqrt_2_over_pi = 0.79788456f;
        const float coeff = 0.044715f;

        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);

        return 0.5f * x * (1.0f + tanh(inner));
    }

    float gelu_der(float x)
    {
        const float kSqrt2OverPi = 0.79788456f;
        const float kCoeff = 0.044715f;

        float x2 = x * x;
        float x3 = x * x2;

        float inner = kSqrt2OverPi * (x + kCoeff * x3);
        float tanh_val = tanh(inner);

        float left = 0.5f * (1.0f + tanh_val);
        float right = 0.5f * x * (1 - tanh_val * tanh_val) *
                      kSqrt2OverPi * (1 + 3 * kCoeff * x2);

        return left + right;
    }

    void gelu(vector<vector<float>>& H)
    {
        for (size_t i = 0; i < H.size(); ++i)
            for (size_t j = 0; j < H[0].size(); ++j)
                H[i][j] = gelu_single(H[i][j]);
    }

    void gelu_derivative(vector<vector<float>>& H)
    {
        for (size_t i = 0; i < H.size(); ++i)
            for (size_t j = 0; j < H[0].size(); ++j)
                H[i][j] = gelu_der(H[i][j]);
    }

    float mean(const vector<float>& v)
    {
        float sum = 0.0f;
        for (float x : v) sum += x;
        return sum / v.size();
    }

    float variance(const vector<float>& v, float m)
    {
        float sum = 0.0f;
        for (float x : v)
            sum += (x - m) * (x - m);

        return sum / v.size();
    }
    
    vector<vector<float>> normalization(const vector<vector<float>>& X)
    {
        const float eps = 1e-5f;

        vector<vector<float>> result(X.size(),
                                     vector<float>(X[0].size()));

        for (size_t i = 0; i < X.size(); ++i)
        {
            float m = mean(X[i]);
            float var = variance(X[i], m);

            float std = sqrt(var + eps);

            for (size_t j = 0; j < X[i].size(); ++j)
                result[i][j] = (X[i][j] - m) / std;
        }

        return result;
    }

    /* ---------------- WEIGHT INITIALIZATION ---------------- */

    vector<vector<vector<float>>> linear(int layers, int cols, int rows)
    {
        random_device rd;
        mt19937 gen(rd());

        float limit = sqrt(6.0f / (rows + cols));
        uniform_real_distribution<float> dist(-limit, limit);

        vector<vector<vector<float>>> values(
            layers,
            vector<vector<float>>(cols, vector<float>(rows)));

        for (int l = 0; l < layers; ++l)
            for (int i = 0; i < cols; ++i)
                for (int j = 0; j < rows; ++j)
                    values[l][i][j] = dist(gen);

        return values;
    }

    vector<vector<vector<float>>> linear(int layers, int dim)
    {
        return linear(layers, dim, dim);
    }

    vector<vector<vector<vector<float>>>> linear3(int layers,
                                                  int heads,
                                                  int rows,
                                                  int cols)
    {
        random_device rd;
        mt19937 gen(rd());

        float limit = sqrt(6.0f / (rows + cols));
        uniform_real_distribution<float> dist(-limit, limit);

        vector<vector<vector<vector<float>>>> values(
            layers,
            vector<vector<vector<float>>>(
                heads,
                vector<vector<float>>(rows,
                vector<float>(cols))));

        for (int l = 0; l < layers; ++l)
            for (int h = 0; h < heads; ++h)
                for (int i = 0; i < rows; ++i)
                    for (int j = 0; j < cols; ++j)
                        values[l][h][i][j] = dist(gen);

        return values;
    }

    vector<vector<float>> linear2(int cols, int rows)
    {
        random_device rd;
        mt19937 gen(rd());

        float limit = sqrt(6.0f / (rows + cols));
        uniform_real_distribution<float> dist(-limit, limit);

        vector<vector<float>> values(cols, vector<float>(rows));

        for (int i = 0; i < cols; ++i)
            for (int j = 0; j < rows; ++j)
                values[i][j] = dist(gen);

        return values;
    }
    
    vector<long long> target_shift(const vector<long long>& tokens)
    {
        vector<long long> target(tokens.size() - 1);

        for (size_t i = 0; i < tokens.size() - 1; ++i)
            target[i] = tokens[i + 1];

        return target;
    }
    
    float loss(const vector<vector<float>>& P,
               const vector<long long>& Y)
    {
        float L = 0.0f;

        for (size_t i = 0; i < P.size(); ++i)
            L += -log(P[i][Y[i]] + 1e-9f);

        return L / P.size();
    }

    vector<vector<float>> gradient(vector<vector<float>> P,
                                   const vector<long long>& Y)
    {
        for (size_t i = 0; i < P.size(); ++i)
            P[i][Y[i]] -= 1.0f;

        return P;
    }

    void update(vector<vector<float>>& W,
                const vector<vector<float>>& dW,
                float lr)
    {
        for (size_t i = 0; i < W.size(); ++i)
            for (size_t j = 0; j < W[0].size(); ++j)
                W[i][j] -= lr * dW[i][j];
    }

    void update(vector<vector<vector<float>>>& W,
                const vector<vector<vector<float>>>& dW,
                float lr)
    {
        for (size_t h = 0; h < W.size(); ++h)
            for (size_t i = 0; i < W[h].size(); ++i)
                for (size_t j = 0; j < W[h][0].size(); ++j)
                    W[h][i][j] -= lr * dW[h][i][j];
    }

    void update(vector<float>& b,
                const vector<float>& db,
                float lr)
    {
        for (size_t i = 0; i < b.size(); ++i)
            b[i] -= lr * db[i];
    }

}

#endif