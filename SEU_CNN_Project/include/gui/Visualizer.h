#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "../core/Tensor.h"
#include "../core/Layer.h"
#include <string>

class Visualizer
{
public:
    Visualizer() = delete;

    static void show_image(const Tensor& image, int batch_index = 0,
                           const std::string& window_name = "image");
    static void show_channel(const Tensor& tensor, int batch_index, int channel_index,
                             const std::string& window_name = "channel");
    static void show_feature_maps(const Tensor& tensor, int batch_index = 0,
                                  const std::string& window_name = "feature maps");
    static void show_layer(Layer& layer, const Tensor& input, int batch_index = 0,
                           const std::string& window_name = "layer output");
};

#endif // VISUALIZER_H
