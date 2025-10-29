//
// Created by tvpower on 10/25/25.
//

#include "inference.h"
#include <iostream>

GoKartInference::GoKartInference(const std::string &model_path)
    : device_(torch::kCUDA) {
    try {
        //load the torchscript model
        model_ = torch::jit::load(model_path);
        model_.to(device_);
        model_.eval();
        std::cout << "Model loaded successfully on GPU" << std::endl;
    } catch (const c10::Error &e) {
        std::cerr << "error loading the model\n" << e.what() << std::endl;
        throw;
    }
}

torch::Tensor GoKartInference::prepcessFrame(const cv::Mat& frame) {
    try {
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(518, 518));

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        cv::Mat float_img;
        rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        // Ensure continuous memory
        if (!float_img.isContinuous()) {
            float_img = float_img.clone();
        }

        auto tensor = torch::from_blob(
            float_img.data,
            {1, 518, 518, 3},
            torch::kFloat32
        ).clone();

        tensor = tensor.permute({0, 3, 1, 2}).contiguous();

        // Create mean and std on CPU first, then move to device with the tensor
        auto mean = torch::tensor({0.485, 0.456, 0.406}, torch::kFloat32).view({1, 3, 1, 1});
        auto std_tensor = torch::tensor({0.229, 0.224, 0.225}, torch::kFloat32).view({1, 3, 1, 1});

        // Normalize on CPU, then move to GPU (more stable)
        tensor = (tensor - mean) / std_tensor;
        tensor = tensor.to(device_);

        return tensor;

    } catch (const std::exception& e) {
        std::cerr << "Error in prepcessFrame: " << e.what() << std::endl;
        throw;
    }
}

Prediction GoKartInference::predict(const cv::Mat& frame) {
    try {
        torch::Tensor input = prepcessFrame(frame);


        std::vector<torch::jit::IValue> inputs;

        auto output = model_.forward(inputs).toTuple();

        //parse outputs
        auto segment_type = output->elements()[0].toTensor();
        auto curve_number = output->elements()[1].toTensor();
        auto direction = output->elements()[2].toTensor();
        auto point_class_tensor = output->elements()[3].toTensor();
        auto coords_tensor = output->elements()[4].toTensor();

        Prediction result;

        // Segment type
        auto segment_probs = torch::softmax(segment_type, 1);
        auto segment_idx = std::get<1>(segment_probs.max(1)).cpu().item<int>();
        std::vector<std::string> segment_labels = {"Curve", "Straight", "Race_Start"};
        result.segment_type = segment_labels[segment_idx];

        // Curve number
        auto curve_idx = std::get<1>(torch::max(curve_number, 1)).cpu().item<int>();
        result.curve_number = curve_idx + 1;

        auto direction_idx = std::get<1>(torch::max(direction, 1)).cpu().item<int>();
        result.direction = direction_idx == 0 ? "Left" : "Right";

        auto point_probs = torch::softmax(point_class_tensor, 1);

        auto point_max_result = torch::max(point_probs, 1);

        auto point_idx_tensor = std::get<1>(point_max_result);

        auto point_idx = point_idx_tensor.cpu().item<int>();

        std::vector<std::string> point_types = {"None", "Turn_in", "Apex", "Exit"};
        result.point_type = point_types[point_idx];

        auto confidence_tensor = point_probs[0][point_idx];

        result.confidence = confidence_tensor.cpu().item<float>();

        auto coords = coords_tensor.cpu();
        result.x_coord = coords[0][0].item<float>();
        result.y_coord = coords[0][1].item<float>();

        return result;

    } catch (const std::exception& e) {
        std::cerr << "    ERROR in predict(): " << e.what() << std::endl;
        throw;
    }
}

void GoKartInference::drawPrediction(cv::Mat &frame, const Prediction &result) {
    int height = frame.rows;
    int width = frame.cols;

    int x = static_cast<int>(result.x_coord * width);
    int y = static_cast<int>(result.y_coord * height);

    cv::Scalar color;
    if (result.point_type == "Turn_in") color = cv::Scalar(0, 255, 0);
    else if (result.point_type == "Apex") color = cv::Scalar(0, 0, 255);
    else color = cv::Scalar(255, 0, 0);

    cv::circle(frame, cv::Point(x, y), 10, color, -1);

    //Text overlay
    std::string text = "Curve " + std::to_string(result.curve_number) + " (" + result.direction + ") - " + result.point_type;
    cv::putText(frame, text, cv::Point(30, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);

    std::string conf_text = "Confidence: " + std::to_string(result.confidence * 100) + "%";
    cv::putText(frame, conf_text, cv::Point(30, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    
}

void GoKartInference::processVideo(const std::string &video_path, const std::string &output_path) {

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return;
    }

    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Video properties: " << frame_width << "x" << frame_height << " @ " << fps << "fps" << std::endl;

    if (frame_width == 0 || frame_height == 0 || fps == 0) {
        std::cerr << "Error reading video properties" << std::endl;
        return;
    }

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps, cv::Size(frame_width, frame_height));

    if (!writer.isOpened()) {
        std::cerr << "Error opening video writer" << std::endl;
        return;
    }

    cv::Mat frame;
    int frame_count = 0;
    while (cap.read(frame)) {
        try {
            std::cout << "Processing frame " << frame_count << "..." << std::endl;

            // Check frame validity
            if (frame.empty()) {
                std::cerr << "Empty frame at " << frame_count << std::endl;
                break;
            }

            std::cout << "  Frame size: " << frame.cols << "x" << frame.rows << std::endl;
            std::cout << "  Calling predict..." << std::endl;

            auto result = predict(frame);

            drawPrediction(frame, result);

            writer.write(frame);

            frame_count++;

        } catch (const std::exception& e) {
            std::cerr << "Error processing frame " << frame_count << ": " << e.what() << std::endl;
            throw;
        }
    }
    cap.release();
    writer.release();
    std::cout << "Video processing complete: " << output_path << std::endl;
}




