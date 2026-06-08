#include"layers/ReluLayer.h"

Tensor ReluLayer::forward(const Tensor& input) {
    // 缓存输入，供 backward 使用
    input_cache_ = input;   // 深拷贝

    int N = input.n();
    int C = input.c();
    int H = input.h();
    int W = input.w();

    // 输出张量，形状与输入相同
    Tensor output(N, C, H, W);

    // 逐元素 ReLU
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float val = input.get(n, c, h, w);
                    if (val > 0.0f) {
                        output.get(n, c, h, w) = val;
                    }
                    else {
                        output.get(n, c, h, w) = 0.0f;
                    }
                }
            }
        }
    }

    return output;
}

Tensor ReluLayer::backward(const Tensor& grad_out) {
    // grad_out 形状应与 input_cache_ 完全一致
    int N = input_cache_.n();
    int C = input_cache_.c();
    int H = input_cache_.h();
    int W = input_cache_.w();

    Tensor grad_input(N, C, H, W);   // 对输入的梯度

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float val = input_cache_.get(n, c, h, w);
                    float grad = grad_out.get(n, c, h, w);
                    // ReLU 的导数为：1 if input > 0 else 0
                    if (val > 0.0f) {
                        grad_input.get(n, c, h, w) = grad;
                    }
                    else {
                        grad_input.get(n, c, h, w) = 0.0f;
                    }
                }
            }
        }
    }

    return grad_input;
}

void ReluLayer::update(float learning_rate) {
    // ReLU 层无可训练参数，空实现
}

void ReluLayer::zero_grad() {
    // 无操作
}

