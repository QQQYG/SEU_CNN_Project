#ifndef LAYER_H
#define LAYER_H

#include "Tensor.h"

// 所有网络层的抽象基类
class Layer {
public:
    virtual ~Layer() = default;

    // 前向传播
    virtual Tensor forward(const Tensor& input) = 0;

    //  反向传播
    virtual Tensor backward(const Tensor& grad_output) = 0;

    // 更新可训练参数（如权重、偏置），使用累积的梯度
    virtual void update(float learning_rate) = 0;



    // 重置梯度（若梯度累加）
    virtual void zero_grad() {}

    // 调试用
    virtual const char* name() const = 0;
};

#endif // LAYER_H