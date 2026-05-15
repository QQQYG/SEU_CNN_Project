#include "utils/MnistLoader.h"
#include <fstream>
#include <iostream>

// 小端转大端 / 大端转小端
int MnistLoader::reverse_int(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

Tensor MnistLoader::load_images(const std::string& filename) {
    std::ifstream file(filename.c_str(), std::ios::binary); 
    if (!file.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        return Tensor(); 
    }

    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    // 读取头部信息 (每次读 4 字节)
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverse_int(number_of_images);

    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);

    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);

    // 构建装载所有图像的巨大 Tensor
    Tensor images(number_of_images, 1, n_rows, n_cols);

    // 逐个像素读取 (MNIST 像素是 0-255 的 unsigned char)
    for (int n = 0; n < number_of_images; ++n) {
        for (int r = 0; r < n_rows; ++r) {
            for (int c = 0; c < n_cols; ++c) {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                // 将 0-255 的像素归一化到 0.0 ~ 1.0
                images.get(n, 0, r, c) = (float)temp / 255.0f;
            }
        }
    }

    file.close();
    return images;
}

Tensor MnistLoader::load_labels(const std::string& filename) {
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Error opening file: " << filename << std::endl;
        return Tensor(); 
    }

    int magic_number = 0;
    int number_of_items = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    if (magic_number != 2049) {
        std::cout << "Warning: Invalid MNIST label magic number: " << magic_number << std::endl;
    }

    file.read((char*)&number_of_items, sizeof(number_of_items));
    number_of_items = reverse_int(number_of_items);
    Tensor labels(number_of_items, 1, 1, 1);

    for (int n = 0; n < number_of_items; ++n) {
        unsigned char temp = 0;
        file.read((char*)&temp, sizeof(temp));

        labels.get(n, 0, 0, 0) = (float)temp;
    }

    file.close();
    return labels;
}