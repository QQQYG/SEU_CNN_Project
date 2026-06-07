#include "../../include/gui/Visualizer.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

void check_tensor(const Tensor& tensor) {
    if (tensor.n() <= 0 || tensor.c() <= 0 || tensor.h() <= 0 || tensor.w() <= 0) {
        throw std::runtime_error("Visualizer: empty tensor");
    }
}

void check_batch_index(const Tensor& tensor, int batch_index) {
    if (batch_index < 0 || batch_index >= tensor.n()) {
        throw std::out_of_range("Visualizer: batch index out of range");
    }
}

void check_channel_index(const Tensor& tensor, int channel_index) {
    if (channel_index < 0 || channel_index >= tensor.c()) {
        throw std::out_of_range("Visualizer: channel index out of range");
    }
}

const float* channel_ptr(const Tensor& tensor, int batch_index, int channel_index) {
    const int plane_size = tensor.h() * tensor.w();
    return tensor.data_ptr() + (batch_index * tensor.c() + channel_index) * plane_size;
}

cv::Mat make_channel_mat(const Tensor& tensor, int batch_index, int channel_index) {
    return cv::Mat(tensor.h(), tensor.w(), CV_32FC1,
                   const_cast<float*>(channel_ptr(tensor, batch_index, channel_index)));
}

cv::Mat float_to_u8(const cv::Mat& source) {
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(source, &min_value, &max_value);

    cv::Mat result;
    if (std::isfinite(min_value) && std::isfinite(max_value) && max_value > min_value) {
        if (min_value >= 0.0 && max_value <= 1.0) {
            source.convertTo(result, CV_8UC1, 255.0);
        } else {
            cv::normalize(source, result, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        }
    } else {
        result = cv::Mat::zeros(source.size(), CV_8UC1);
    }

    return result;
}

cv::Mat make_image_mat(const Tensor& tensor, int batch_index) {
    if (tensor.c() == 1) {
        return float_to_u8(make_channel_mat(tensor, batch_index, 0));
    }

    if (tensor.c() == 3) {
        std::vector<cv::Mat> channels;
        channels.reserve(3);
        for (int channel = 0; channel < 3; ++channel) {
            channels.push_back(float_to_u8(make_channel_mat(tensor, batch_index, channel)));
        }

        cv::Mat image;
        cv::merge(channels, image);
        return image;
    }

    throw std::runtime_error("Visualizer: show_image supports only 1-channel or 3-channel tensors");
}

cv::Mat make_feature_grid(const Tensor& tensor, int batch_index) {
    const int channel_count = tensor.c();
    const int columns = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(channel_count))));
    const int rows = (channel_count + columns - 1) / columns;

    cv::Mat grid = cv::Mat::zeros(rows * tensor.h(), columns * tensor.w(), CV_8UC1);

    for (int channel = 0; channel < channel_count; ++channel) {
        cv::Mat feature = float_to_u8(make_channel_mat(tensor, batch_index, channel));
        const int row = channel / columns;
        const int column = channel % columns;
        cv::Rect target(column * tensor.w(), row * tensor.h(), tensor.w(), tensor.h());
        feature.copyTo(grid(target));
    }

    return grid;
}

void show_mat(const std::string& window_name, const cv::Mat& mat) {
    cv::imshow(window_name, mat);
    cv::waitKey(0);
}

} 

void Visualizer::show_image(const Tensor& image, int batch_index,
                            const std::string& window_name) {
    check_tensor(image);
    check_batch_index(image, batch_index);

    show_mat(window_name, make_image_mat(image, batch_index));
}

void Visualizer::show_channel(const Tensor& tensor, int batch_index, int channel_index,
                              const std::string& window_name) {
    check_tensor(tensor);
    check_batch_index(tensor, batch_index);
    check_channel_index(tensor, channel_index);

    show_mat(window_name, float_to_u8(make_channel_mat(tensor, batch_index, channel_index)));
}

void Visualizer::show_feature_maps(const Tensor& tensor, int batch_index,
                                   const std::string& window_name) {
    check_tensor(tensor);
    check_batch_index(tensor, batch_index);

    show_mat(window_name, make_feature_grid(tensor, batch_index));
}

void Visualizer::show_layer(Layer& layer, const Tensor& input, int batch_index,
                            const std::string& window_name) {
    Tensor output = layer.forward(input);
    show_feature_maps(output, batch_index, window_name);
}
