#pragma once
#include "core/Tensor.h"
#include "core/Layer.h"
#include <vector>

class PoolLayer : public Layer {
public:
    PoolLayer(int pool_size, int stride = 2, bool max_pool = true);
    virtual ~PoolLayer() = default;

    Tensor forward(const Tensor& input);
    Tensor backward(const Tensor& grad_output);
    void update(float lr);

private:
    int pool_h_, pool_w_;
    int stride_h_, stride_w_;
    bool max_pool_;
    Tensor cached_input_;
    int out_h_, out_w_;
    std::vector<int> max_index_;
}; 
