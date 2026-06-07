#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "core/Layer.h"
#include "core/Tensor.h"

class DenseLayer : public Layer {
private:
    int in_features_;   // 输入特征数 = C * H * W
    int out_features_;

    // 可训练参数
    Tensor weights_;   
    Tensor bias_;       

    // 梯度（与参数形状相同）
    Tensor grad_weights_;
    Tensor grad_bias_;

    // 前向缓存：保存展平后的输入矩阵
    Tensor input_cache_;
public:
    // 构造函数：in_features 输入特征数（C*H*W），out_features 输出特征数
    DenseLayer(int in_features, int out_features);
    virtual ~DenseLayer() = default;

    // 前向传播
    Tensor forward(const Tensor& input) override;

    // 反向传播
    Tensor backward(const Tensor& grad_output) override;

    // 更新权重和偏置
    void update(float learning_rate) override;

 

    // 重置梯度（每个 batch 前可选调用）
    void zero_grad() override;

    // 层名称
    const char* name() const override { return "DenseLayer"; }

    // 获取权重/偏置（用于可视化或保存）
    Tensor get_weights() const { return weights_; }
    Tensor get_bias() const { return bias_; }

};

#endif 