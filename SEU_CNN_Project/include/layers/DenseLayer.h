#ifndef DENSE_LAYER_H       
#define DENSE_LAYER_H

#include "core/Layer.h"
#include "core/Tensor.h"

class DenseLayer : public Layer {
private:
    int in_features_;   // ���������� = C * H * W
    int out_features_;

    // ��ѵ������
    Tensor weights_;   
    Tensor bias_;       

    // �ݶȣ��������״��ͬ��
    Tensor grad_weights_;
    Tensor grad_bias_;

    // ǰ�򻺴棺����չƽ����������
    Tensor cached_flat_input_;
public:
    // ���캯����in_features ������������C*H*W����out_features ���������
    DenseLayer(int in_features, int out_features);
    virtual ~DenseLayer() = default;

    // ǰ�򴫲�
    Tensor forward(const Tensor& input) override;

    // ���򴫲�
    Tensor backward(const Tensor& grad_output) override;

    // ����Ȩ�غ�ƫ��
    void update(float learning_rate) override;

 

    // �����ݶȣ�ÿ�� batch ǰ��ѡ���ã�
    void zero_grad() override;

    // ������
    const char* name() const override { return "DenseLayer"; }

    // ��ȡȨ��/ƫ�ã����ڿ��ӻ��򱣴棩
    Tensor get_weights() const { return weights_; }
    Tensor get_bias() const { return bias_; }

};

#endif 