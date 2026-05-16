#pragma once
#include "core/Tensor.h"
#include "core/Layer.h"

class ConvLayer : public Layer {
public:
    ConvLayer(int in_channels, int out_channels, int kernel_size,
        int stride = 1, int padding = 0, bool use_xavier = true);
    virtual ~ConvLayer() = default;

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    void update(float lr);

private:
    int in_channels_, out_channels_;
    int kernel_h_, kernel_w_;
    int stride_h_, stride_w_;
    int pad_h_, pad_w_;
    bool use_xavier_;

    Tensor weight_;        // [out_c, in_c, kh, kw]
    Tensor bias_;          // [out_c]
    Tensor grad_weight_;
    Tensor grad_bias_;

    Tensor cached_input_;
    Tensor padded_input_;
    int out_h_, out_w_;

    Tensor applyPadding(const Tensor& input, int pad_h, int pad_w) const;
    void initWeights();
    void computeOutputDims(int in_h, int in_w, int& out_h, int& out_w) const;
};