#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor {
private:
    int batch_size_;
    int channels_;
    int height_;
    int width_;
    std::vector<float> data_v; // 使用 vector 管理底层内存，自动处理析构，但要注意深拷贝

public:
    // 默认构造
    Tensor();

    // 带维度的构造函数 (自动分配大小为 N*C*H*W 的全 0 数组)
    Tensor(int n, int c, int h, int w);

    // 获取维度信息
    int n() const { return batch_size_; }
    int c() const { return channels_; }
    int h() const { return height_; }
    int w() const { return width_; }
    int size() const { return batch_size_ * channels_ * height_ * width_; }

    // 获取底层数据指针（给 D 同学可视化用的接口）
    float* data_ptr();
    const float* data_ptr() const;

    // 核心算法：四维坐标转一维索引
    int get_index(int n, int c, int h, int w) const;

    // 读写特定位置的元素 (返回引用，允许修改)
    float& get(int n, int c, int h, int w);
    const float& get(int n, int c, int h, int w) const;

    // 矩阵初始化工具
    void fill_random(); // 填充随机数 (给权重初始化用)
    void fill_zero();   // 填充 0
};

#endif // TENSOR_H