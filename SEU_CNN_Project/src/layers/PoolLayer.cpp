    #include "layers/PoolLayer.h"
#include <algorithm>
#include <stdexcept>

PoolLayer::PoolLayer(int pool_size, int stride, bool max_pool)
    : pool_h_(pool_size), pool_w_(pool_size),
    stride_h_(stride), stride_w_(stride), max_pool_(max_pool) {
    if (!max_pool_) {
        throw std::runtime_error("PoolLayer: only max pooling is implemented");
    }
}

Tensor PoolLayer::forward(const Tensor& input) {
    cached_input_ = input;
    int n = input.n(), c = input.c();
    int in_h = input.h(), in_w = input.w();
    out_h_ = (in_h - pool_h_) / stride_h_ + 1;
    out_w_ = (in_w - pool_w_) / stride_w_ + 1;
    if (out_h_ <= 0 || out_w_ <= 0)
        throw std::runtime_error("PoolLayer: invalid dimensions");

    Tensor output(n, c, out_h_, out_w_);
    output.fill_zero();
    float* out_data = output.data_ptr();
    const float* in_data = input.data_ptr();

    max_index_.clear();
    max_index_.resize(n * c * out_h_ * out_w_);

    int idx = 0;
    for (int ni = 0; ni < n; ++ni) {
        for (int ci = 0; ci < c; ++ci) {
            for (int oh = 0; oh < out_h_; ++oh) {
                int start_h = oh * stride_h_;
                for (int ow = 0; ow < out_w_; ++ow) {
                    int start_w = ow * stride_w_;
                    float max_val = -1e38f;
                    int max_pos = -1;
                    for (int kh = 0; kh < pool_h_; ++kh) {
                        int in_h_idx = start_h + kh;
                        for (int kw = 0; kw < pool_w_; ++kw) {
                            int in_w_idx = start_w + kw;
                            int src_idx = ((ni * c + ci) * in_h + in_h_idx) * in_w + in_w_idx;
                            float val = in_data[src_idx];
                            if (val > max_val) {
                                max_val = val;
                                max_pos = src_idx;
                            }
                        }
                    }
                    out_data[idx] = max_val;
                    max_index_[idx] = max_pos;
                    ++idx;
                }
            }
        }
    }
    return output;
}

Tensor PoolLayer::backward(const Tensor& grad_output) {
    Tensor grad_input(cached_input_.n(), cached_input_.c(),
        cached_input_.h(), cached_input_.w());
    grad_input.fill_zero();
    float* grad_in_data = grad_input.data_ptr();
    const float* grad_out_data = grad_output.data_ptr();

    int idx = 0;
    for (int ni = 0; ni < cached_input_.n(); ++ni) {
        for (int ci = 0; ci < cached_input_.c(); ++ci) {
            for (int oh = 0; oh < out_h_; ++oh) {
                for (int ow = 0; ow < out_w_; ++ow) {
                    float grad_val = grad_out_data[idx];
                    int max_pos = max_index_[idx];
                    grad_in_data[max_pos] += grad_val;
                    ++idx;
                }
            }
        }
    }
    return grad_input;
}

void PoolLayer::update(float /*lr*/) {
    // 池化层无可训练参数
}