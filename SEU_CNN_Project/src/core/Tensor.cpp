#include "core/Tensor.h"
#include <cstdlib> 

// 注意：这里绝对不能有 { 

Tensor::Tensor() : batch_size_(0), channels_(0), height_(0), width_(0) {}

Tensor::Tensor(int n, int c, int h, int w)
    : batch_size_(n), channels_(c), height_(h), width_(w) {
    data_v.assign(n * c * h * w, 0.0f);
}

float* Tensor::data_ptr() {
    return data_v.data();
}

const float* Tensor::data_ptr() const {
    return data_v.data();
}

int Tensor::get_index(int n, int c, int h, int w) const {
    // 这里顺便帮你修掉了一个多余的分号 ;;
    return ((n * channels_ + c) * height_ + h) * width_ + w;
}

float& Tensor::get(int n, int c, int h, int w) {
    return data_v[get_index(n, c, h, w)];
}

const float& Tensor::get(int n, int c, int h, int w) const {
    return data_v[get_index(n, c, h, w)];
}

void Tensor::fill_random() {
    for (size_t i = 0; i < data_v.size(); ++i) {
        data_v[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }
}

void Tensor::fill_zero() {
    for (size_t i = 0; i < data_v.size(); ++i) {
        data_v[i] = 0.0f;
    }
}
// 注意：文件末尾也不要有对应开头那个多余的 }