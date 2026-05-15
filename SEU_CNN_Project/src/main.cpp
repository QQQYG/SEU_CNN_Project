#include <iostream>
#include "utils/MnistLoader.h"

int main() {
    std::cout << "Loading MNIST data..." << std::endl;

    // 加载训练集
    Tensor train_images = MnistLoader::load_images("data/train-images.idx3-ubyte");
    Tensor train_labels = MnistLoader::load_labels("data/train-labels.idx1-ubyte");

    // 验证数量是否匹配
    if (train_images.n() == 60000 && train_labels.n() == 60000) {
        std::cout << "Data loaded successfully!" << std::endl;

        // 打印第一个样本验证一下
        std::cout << "The first label is: " << train_labels.get(0, 0, 0, 0) << std::endl;

        // 打印第一张图片中心点的一个像素看看是不是在 0~1 之间
        std::cout << "A center pixel of the first image is: "
            << train_images.get(0, 0, 14, 14) << std::endl;
    }
    else {
        std::cout << "Data loading failed or paths are incorrect." << std::endl;
    }

    return 0;
}