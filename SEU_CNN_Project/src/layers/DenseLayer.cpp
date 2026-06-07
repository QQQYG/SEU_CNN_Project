#include"layers/DenseLayer.h"
#include <cmath>
#include <cstring>

DenseLayer::DenseLayer(int in_features, int out_features)
	:in_features_(in_features), out_features_(out_features) {

	weights_ = Tensor(out_features_, in_features_, 1, 1);
	bias_ = Tensor(1, out_features, 1, 1);

	weights_.fill_random();
	bias_.fill_zero();//初始化

	grad_weights_ = Tensor(out_features, in_features, 1, 1);
	grad_bias_ = Tensor(1, out_features, 1, 1);

	grad_weights_.fill_zero();
	grad_bias_.fill_zero();
}

Tensor DenseLayer::forward(const Tensor& input) {
	int N = input.n();          // batch size
	int C = input.c();
	int H = input.h();
	int W = input.w();
	int spatial = C * H * W;    // 输入特征数

	input_cache_ = input;

	Tensor output(N, 1, 1, out_features_);

	// 矩阵乘法：output[n][0][0][o] = sum_i( input[n][c][h][w] * weights_[o][i][0][0] ) + bias_[0][o][0][0]
	for (int n = 0; n < N; ++n) {
		for (int o = 0; o < out_features_; ++o) {
			float sum = bias_.get(0, o, 0, 0);  // 先加上偏置
			// 遍历所有空间位置，i = c*H*W + h*W + w 线性索引
			int i = 0;
			for (int c = 0; c < C; ++c) {
				for (int h = 0; h < H; ++h) {
					for (int w = 0; w < W; ++w) {
						sum += input.get(n, c, h, w) * weights_.get(o, i, 0, 0);
						i++;
					}
				}
			}
			output.get(n, 0, 0, o) = sum;
		}
	}
	return output;
}

Tensor DenseLayer::backward(const Tensor& grad_output) {
	// grad_output 形状 (N, 1, 1, out_features_)
	int N = grad_output.n();
	int C = input_cache_.c();
	int H = input_cache_.h();
	int W = input_cache_.w();
	int spatial = C * H * W;

	// 1. 计算对输入的梯度grad_input
	Tensor grad_input(N, C, H, W);
	grad_input.fill_zero();

	// grad_input = grad_output * weights_^T
	// 即对每个样本 n 和每个空间位置 i，grad_input[n][i] = sum_o (grad_output[n][o] * weights_[o][i])
	for (int n = 0; n < N; ++n) {
		int i = 0;
		for (int c = 0; c < C; ++c) {
			for (int h = 0; h < H; ++h) {
				for (int w = 0; w < W; ++w) {
					float val = 0.0f;
					for (int o = 0; o < out_features_; ++o) {
						val += grad_output.get(n, 0, 0, o) * weights_.get(o, i, 0, 0);
					}
					grad_input.get(n, c, h, w) = val;
					i++;
				}
			}
		}
	}

	// 2. 累加权重的梯度（grad_weights_）
    // dW[o][i] = sum_n grad_output[n][o] * input_cache_[n][i]
	for (int o = 0; o < out_features_; ++o) {
		for (int i = 0; i < in_features_; ++i) {
			float sum = 0.0f;
			for (int n = 0; n < N; ++n) {
				// 将二维索引 i 映射回四维坐标
				int c = i / (H * W);
				int h = (i / W) % H;
				int w = i % W;
				sum += grad_output.get(n, 0, 0, o) * input_cache_.get(n, c, h, w);
			}
			grad_weights_.get(o, i, 0, 0) += sum;
		}
	}

	// 3. 累加偏置的梯度（grad_bias_）
	// db[0][o] = sum_n grad_output[n][o]
	for (int o = 0; o < out_features_; ++o) {
		float sum = 0.0f;
		for (int n = 0; n < N; ++n) {
			sum += grad_output.get(n, 0, 0, o);
		}
		grad_bias_.get(0, o, 0, 0) += sum;
	}

	return grad_input;
}

void DenseLayer::update(float learning_rate) {
	// weights_ -= learning_rate * grad_weights_
	for (int o = 0; o < out_features_; ++o) {
		for (int i = 0; i < in_features_; ++i) {
			float& w = weights_.get(o, i, 0, 0);
			float gw = grad_weights_.get(o, i, 0, 0);
			w -= learning_rate * gw;
		}
	}

	// bias_ -= learning_rate * grad_bias_
	for (int o = 0; o < out_features_; ++o) {
		float& b = bias_.get(0, o, 0, 0);
		float gb = grad_bias_.get(0, o, 0, 0);
		b -= learning_rate * gb;
	}

	zero_grad();
}

void DenseLayer::zero_grad() {
	grad_weights_.fill_zero();
	grad_bias_.fill_zero();
}

