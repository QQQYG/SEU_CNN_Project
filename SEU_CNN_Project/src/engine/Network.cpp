#include "Layers/Network.h"
#include <cmath>
#include <iostream>
#include <algorithm>  

Network::Network() : learning_rate_(0.01f) {}

//把一层网络加入到当前神经网络中。
void Network::add(Layer* layer) {
    layers_.push_back(layer);
}

float Network::softmax_cross_entropy(const Tensor& logits, const Tensor& labels, Tensor& dlogits) {
    // logits: (N, 1, 1, num_classes)
    // labels: (N, 1, 1, 1)
    int N = logits.n();
    int num_classes = logits.w();   

    // 输出梯度，形状与 logits 相同，初始化为 softmax 概率
    dlogits = Tensor(N, 1, 1, num_classes);
    float loss = 0.0f;

    for (int n = 0; n < N; ++n) {
        // 1. 获取最大值
        float max_val = logits.get(n, 0, 0, 0);
        for (int k = 1; k < num_classes; ++k) {
            float v = logits.get(n, 0, 0, k);
            if (v > max_val) max_val = v;
        }

        // 2. 计算 exp 及总和
        float sum_exp = 0.0f;
        std::vector<float> exp_vals(num_classes);
        for (int k = 0; k < num_classes; ++k) {
            float i= std::exp(logits.get(n, 0, 0, k) - max_val);
            exp_vals[k] = i;
            sum_exp += i;
        }

        // 3. Softmax 概率 + 交叉熵损失
        int label = static_cast<int>(labels.get(n, 0, 0, 0));  
        for (int k = 0; k < num_classes; ++k) {
            float prob = exp_vals[k] / sum_exp;
            dlogits.get(n, 0, 0, k) = prob;   // 暂时存概率
        }
        loss -= std::log(dlogits.get(n, 0, 0, label));   // 交叉熵

        // 4. 梯度 = softmax - one_hot
        for (int k = 0; k < num_classes; ++k) {
            float one_hot = (k == label) ? 1.0f : 0.0f;
            dlogits.get(n, 0, 0, k) -= one_hot;
        }
    }

    loss /= N;   // 平均损失
    return loss;
}

void Network::train(const Tensor& inputs, const Tensor& labels, int epochs, int batch_size, float lr) {
    learning_rate_ = lr;
    int N = inputs.n();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;

        for (int start = 0; start < N; start += batch_size) {
            int end = std::min(start + batch_size, N);  //防止越界
            int current_batch_size = end - start;

            // 构造 batch（复制数据
            Tensor batch_inputs(current_batch_size, inputs.c(), inputs.h(), inputs.w());
            Tensor batch_labels(current_batch_size, 1, 1, 1);

            for (int i = 0; i < current_batch_size; ++i) {
                int src_idx = start + i;
                // 复制图像
                for (int c = 0; c < inputs.c(); ++c)
                    for (int h = 0; h < inputs.h(); ++h)
                        for (int w = 0; w < inputs.w(); ++w)
                            batch_inputs.get(i, c, h, w) = inputs.get(src_idx, c, h, w);
                // 复制标签
                batch_labels.get(i, 0, 0, 0) = labels.get(src_idx, 0, 0, 0);
            }

            // 前向传播
            Tensor output = layers_[0]->forward(batch_inputs);
            for (size_t i = 1; i < layers_.size(); ++i) {
                output = layers_[i]->forward(output);
            }

            // 计算损失 + 得到输出梯度
            Tensor grad;
            float loss = softmax_cross_entropy(output, batch_labels, grad);
            epoch_loss += loss;
            num_batches++;

            // 反向传播 
            for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
                grad = layers_[i]->backward(grad);
            }

            
            for (auto* layer : layers_) {
                layer->update(learning_rate_);
            }

            // 安全起见，再清一次梯度（有些层 update 里已清零）
            for (auto* layer : layers_) {
                layer->zero_grad();
            }
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs
            << " - Loss: " << epoch_loss / num_batches << std::endl;
    }
}

float Network::evaluate(const Tensor& inputs, const Tensor& labels) {
    int N = inputs.n();

    // 一次前向传播得到整个测试集的输出
    Tensor output = inputs;
    for (size_t i = 0; i < layers_.size(); ++i) {
        output = layers_[i]->forward(output);
    }
    // out 形状: (N, 1, 1, num_classes)

    int correct = 0;
    int num_classes = output.w();

    for (int n = 0; n < N; ++n) {
        // 寻找第 n 个样本预测类别
        int pred = 0;
        float max_val = output.get(n, 0, 0, 0);
        for (int k = 1; k < num_classes; ++k) {
            float val = output.get(n, 0, 0, k);
            if (val > max_val) {
                max_val = val;
                pred = k;
            }
        }

        int label = static_cast<int>(labels.get(n, 0, 0, 0));
        if (pred == label) correct++;
    }

    return static_cast<float>(correct) / N;
}
