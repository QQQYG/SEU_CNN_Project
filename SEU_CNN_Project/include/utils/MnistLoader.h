#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include "core/Tensor.h"
#include <string>

class MnistLoader {
private:
    // 核心算法：翻转 32 位整数的字节序
    static int reverse_int(int i);

public:
    // 读取图像文件，返回 Tensor (维度: batch_size, 1, 28, 28)
    static Tensor load_images(const std::string& filename);

    // 读取标签文件，返回 Tensor (维度: batch_size, 1, 1, 1)
    static Tensor load_labels(const std::string& filename);
};

#endif // MNIST_LOADER_H