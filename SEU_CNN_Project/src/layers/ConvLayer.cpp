#include "layers/ConvLayer.h"
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>
#include <stdexcept>

static std::mt19937& getRng() {
    static thread_local std::mt19937 rng(std::random_device{}());
    return rng;
}

ConvLayer::ConvLayer(int in_channels, int out_channels, int kernel_size,
    int stride, int padding, bool use_xavier)
    : in_channels_(in_channels), out_channels_(out_channels),
    kernel_h_(kernel_size), kernel_w_(kernel_size),
    stride_h_(stride), stride_w_(stride),
    pad_h_(padding), pad_w_(padding), use_xavier_(use_xavier) {
    weight_ = Tensor(out_channels_, in_channels_, kernel_h_, kernel_w_);
    bias_ = Tensor(out_channels_, 1, 1, 1);  // 偏置只存储 out_channels 个标量，但 Tensor 必须是 4D
    // 为了简化，偏置视为形状 [out_c, 1, 1, 1]
    grad_weight_ = Tensor(out_channels_, in_channels_, kernel_h_, kernel_w_);
    grad_bias_ = Tensor(out_channels_, 1, 1, 1);
    initWeights();
}

void ConvLayer::initWeights() {
    int fan_in = in_channels_ * kernel_h_ * kernel_w_;
    int fan_out = out_channels_ * kernel_h_ * kernel_w_;
    float limit = 0.0f;
    if (use_xavier_) {
        limit = std::sqrt(6.0f / (fan_in + fan_out));
    }
    else {
        limit = std::sqrt(2.0f / fan_in);
    }
    std::uniform_real_distribution<float> dist(-limit, limit);
    float* w_data = weight_.data_ptr();
    for (int i = 0; i < weight_.size(); ++i) {
        w_data[i] = dist(getRng());
    }
    bias_.fill_zero();
    grad_weight_.fill_zero();
    grad_bias_.fill_zero();
}

void ConvLayer::computeOutputDims(int in_h, int in_w, int& out_h, int& out_w) const {
    out_h = (in_h + 2 * pad_h_ - kernel_h_) / stride_h_ + 1;
    out_w = (in_w + 2 * pad_w_ - kernel_w_) / stride_w_ + 1;
}

Tensor ConvLayer::applyPadding(const Tensor& input, int pad_h, int pad_w) const {
    int n = input.n(), c = input.c(), h = input.h(), w = input.w();
    int new_h = h + 2 * pad_h, new_w = w + 2 * pad_w;
    Tensor padded(n, c, new_h, new_w);
    padded.fill_zero();

    const float* src = input.data_ptr();
    float* dst = padded.data_ptr();
    int src_line = w;
    int dst_line = new_w;
    int src_plane = h * w;
    int dst_plane = new_h * new_w;

    for (int ni = 0; ni < n; ++ni) {
        for (int ci = 0; ci < c; ++ci) {
            const float* src_slice = src + (ni * c + ci) * src_plane;
            float* dst_slice = dst + (ni * c + ci) * dst_plane;
            // 跳过顶部 padding 行
            dst_slice += pad_h * dst_line + pad_w;
            for (int hi = 0; hi < h; ++hi) {
                memcpy(dst_slice, src_slice, w * sizeof(float));
                src_slice += src_line;
                dst_slice += dst_line;
            }
        }
    }
    return padded;
}

Tensor ConvLayer::forward(const Tensor& input) {
    cached_input_ = input;
    int n = input.n(), c = input.c();
    int in_h = input.h(), in_w = input.w();
    if (c != in_channels_) throw std::runtime_error("ConvLayer: input channel mismatch");
    computeOutputDims(in_h, in_w, out_h_, out_w_);
    padded_input_ = applyPadding(input, pad_h_, pad_w_);

    const float* x_data = padded_input_.data_ptr();
    int padded_h = padded_input_.h(), padded_w = padded_input_.w();
    Tensor output(n, out_channels_, out_h_, out_w_);
    output.fill_zero();
    float* out_data = output.data_ptr();
    const float* w_data = weight_.data_ptr();
    const float* b_data = bias_.data_ptr();

    for (int ni = 0; ni < n; ++ni) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            float bias_val = b_data[oc];
            for (int oh = 0; oh < out_h_; ++oh) {
                int start_h = oh * stride_h_;
                for (int ow = 0; ow < out_w_; ++ow) {
                    int start_w = ow * stride_w_;
                    float sum = bias_val;
                    for (int ic = 0; ic < in_channels_; ++ic) {
                        for (int kh = 0; kh < kernel_h_; ++kh) {
                            int in_h_idx = start_h + kh;
                            for (int kw = 0; kw < kernel_w_; ++kw) {
                                int in_w_idx = start_w + kw;
                                int x_idx = ((ni * in_channels_ + ic) * padded_h + in_h_idx) * padded_w + in_w_idx;
                                int w_idx = ((oc * in_channels_ + ic) * kernel_h_ + kh) * kernel_w_ + kw;
                                sum += x_data[x_idx] * w_data[w_idx];
                            }
                        }
                    }
                    int out_idx = ((ni * out_channels_ + oc) * out_h_ + oh) * out_w_ + ow;
                    out_data[out_idx] = sum;
                }
            }
        }
    }
    return output;
}

Tensor ConvLayer::backward(const Tensor& grad_output) {
    int n = grad_output.n();
    if (n != cached_input_.n() || grad_output.c() != out_channels_ ||
        grad_output.h() != out_h_ || grad_output.w() != out_w_)
        throw std::runtime_error("ConvLayer: grad_output shape mismatch");

    grad_weight_.fill_zero();
    grad_bias_.fill_zero();

    int orig_h = cached_input_.h(), orig_w = cached_input_.w();
    Tensor grad_input(cached_input_.n(), cached_input_.c(), orig_h, orig_w);
    grad_input.fill_zero();

    int padded_h = padded_input_.h(), padded_w = padded_input_.w();
    Tensor grad_padded(cached_input_.n(), in_channels_, padded_h, padded_w);
    grad_padded.fill_zero();

    const float* x_data = padded_input_.data_ptr();
    const float* grad_out_data = grad_output.data_ptr();
    const float* w_data = weight_.data_ptr();
    float* grad_w_data = grad_weight_.data_ptr();
    float* grad_b_data = grad_bias_.data_ptr();
    float* grad_pad_data = grad_padded.data_ptr();

    // 1. 计算权重和偏置梯度
    for (int ni = 0; ni < n; ++ni) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int oh = 0; oh < out_h_; ++oh) {
                int start_h = oh * stride_h_;
                for (int ow = 0; ow < out_w_; ++ow) {
                    int start_w = ow * stride_w_;
                    float grad_val = grad_out_data[((ni * out_channels_ + oc) * out_h_ + oh) * out_w_ + ow];
                    grad_b_data[oc] += grad_val;
                    for (int ic = 0; ic < in_channels_; ++ic) {
                        for (int kh = 0; kh < kernel_h_; ++kh) {
                            int in_h = start_h + kh;
                            for (int kw = 0; kw < kernel_w_; ++kw) {
                                int in_w = start_w + kw;
                                int x_idx = ((ni * in_channels_ + ic) * padded_h + in_h) * padded_w + in_w;
                                int w_idx = ((oc * in_channels_ + ic) * kernel_h_ + kh) * kernel_w_ + kw;
                                grad_w_data[w_idx] += x_data[x_idx] * grad_val;
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. 计算对输入的梯度（通过 padding 后的梯度图）
    for (int ni = 0; ni < n; ++ni) {
        for (int ic = 0; ic < in_channels_; ++ic) {
            for (int in_h = 0; in_h < padded_h; ++in_h) {
                for (int in_w = 0; in_w < padded_w; ++in_w) {
                    float sum = 0.0f;
                    for (int oc = 0; oc < out_channels_; ++oc) {
                        for (int kh = 0; kh < kernel_h_; ++kh) {
                            int oh_candidate = in_h - kh;
                            if (oh_candidate % stride_h_ != 0) continue;
                            int oh = oh_candidate / stride_h_;
                            if (oh < 0 || oh >= out_h_) continue;
                            for (int kw = 0; kw < kernel_w_; ++kw) {
                                int ow_candidate = in_w - kw;
                                if (ow_candidate % stride_w_ != 0) continue;
                                int ow = ow_candidate / stride_w_;
                                if (ow < 0 || ow >= out_w_) continue;
                                float grad_out = grad_out_data[((ni * out_channels_ + oc) * out_h_ + oh) * out_w_ + ow];
                                int w_idx = ((oc * in_channels_ + ic) * kernel_h_ + kh) * kernel_w_ + kw;
                                sum += grad_out * w_data[w_idx];
                            }
                        }
                    }
                    int grad_idx = ((ni * in_channels_ + ic) * padded_h + in_h) * padded_w + in_w;
                    grad_pad_data[grad_idx] = sum;
                }
            }
        }
    }

    // 裁剪 padding 区域到原始尺寸
    for (int ni = 0; ni < n; ++ni) {
        for (int ic = 0; ic < in_channels_; ++ic) {
            for (int hi = 0; hi < orig_h; ++hi) {
                int padded_hi = hi + pad_h_;
                for (int wi = 0; wi < orig_w; ++wi) {
                    int padded_wi = wi + pad_w_;
                    int src_idx = ((ni * in_channels_ + ic) * padded_h + padded_hi) * padded_w + padded_wi;
                    int dst_idx = ((ni * in_channels_ + ic) * orig_h + hi) * orig_w + wi;
                    grad_input.data_ptr()[dst_idx] = grad_pad_data[src_idx];
                }
            }
        }
    }
    return grad_input;
}

void ConvLayer::update(float lr) {
    float* w_data = weight_.data_ptr();
    float* gw_data = grad_weight_.data_ptr();
    for (int i = 0; i < weight_.size(); ++i) {
        w_data[i] -= lr * gw_data[i];
    }
    float* b_data = bias_.data_ptr();
    float* gb_data = grad_bias_.data_ptr();
    for (int i = 0; i < bias_.size(); ++i) {
        b_data[i] -= lr * gb_data[i];
    }
    grad_weight_.fill_zero();
    grad_bias_.fill_zero();
}