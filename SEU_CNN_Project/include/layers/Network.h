#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include "core/Layer.h"
#include "core/Tensor.h"

class Network {
private:
    std::vector<Layer*> layers_;
    float learning_rate_;

    // 数值稳定的 Softmax + 交叉熵，返回损失值，输出梯度dlogits
    float softmax_cross_entropy(const Tensor& logits, const Tensor& labels, Tensor& dlogits);

public:
    Network();

    // 添加层（按顺序）
    void add(Layer* layer);

    
    // inputs: 形状 (N, C, H, W) 的训练图像
    // labels: 形状 (N, 1, 1, 1)，每个元素是类别索引（0~9）
    void train(const Tensor& inputs, const Tensor& labels, int epochs, int batch_size, float lr);



    // 测试集准确率
    float evaluate(const Tensor& inputs, const Tensor& labels);
};

#endif
