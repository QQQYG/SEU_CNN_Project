#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "core/Layer.h"
#include "core/Tensor.h"

class ReluLayer : public Layer {
private:
 
    // 这里声明一个缓存张量，用于存储输入 > 0 的标志（可存储为 float 0/1 或单独 vector）
    Tensor  input_cache_;   // 形状与 input 相同，存储 1.0f 表示正值，0.0f 表示非正值
public:
 //   ReluLayer() = default;
    virtual ~ReluLayer() = default;

    // 前向传播
    Tensor forward(const Tensor& input) override;

    // 反向传播
    Tensor backward(const Tensor& grad_output) override;

    // 没有可训练参数，无需更新
    void update(float learning_rate) override {};


    // 重置梯度（无参数，无需操作）
    void zero_grad() override {};

    // 层名称
  //  const char* name() const override { return "ReLU"; }


};

#endif
