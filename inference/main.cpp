//
// Created by tvpower on 10/25/25.
//
#include <torch/torch.h>
#include <iostream>
#include "inference.h"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./gokart_inference <model_path> <input_video> <output_video>"<< std::endl;
        std::cout << "model " << argv[2] << std::endl;
        return -1;
    }
    std::string model_path = argv[1];
    std::string input_video = argv[2];
    std::string output_video = argv[3];

    try {
        GoKartInference inference(model_path);

        inference.processVideo(input_video, output_video);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }


    return 0;
}

