//
// Created by tvpower on 10/25/25.
//

#ifndef INFERENCE_INFERENCE_H
#define INFERENCE_INFERENCE_H

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Prediction {
    std::string segment_type;
    int curve_number;
    std::string direction;
    std::string point_type;
    float x_coord;
    float y_coord;
    float confidence;
};

class GoKartInference {
public:
    GoKartInference(const std::string& model_path);
    Prediction predict(const cv::Mat& frame);
    void processVideo(const std::string& video_path, const std::string& output_path);

private:
    torch::jit::script::Module model_;
    torch::Device device_;

    torch::Tensor prepcessFrame(const cv::Mat& frame);
    void drawPrediction(cv::Mat& frame, const Prediction& result);

};
#endif //INFERENCE_INFERENCE_H