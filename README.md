# Go-Kart Racing Coach

A computer vision system for analyzing go-kart racing lines and identifying critical racing points using deep learning. The system detects track segments, curve numbers, racing line directions, and key racing points (turn-in, apex, exit) with pixel-level coordinate prediction.

## Overview

This project combines a frozen DINOv2 vision transformer backbone with a custom multi-head architecture to analyze racing footage. It processes video frames to identify track characteristics and optimal racing points, providing real-time coaching feedback.

## Model Architecture

### Backbone
- **Base Model**: DINOv2 ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`)
- **Feature Dimension**: 1024-dimensional CLS token embeddings
- **Training Strategy**: Frozen backbone (transfer learning)
- **Input Size**: 224x224 (training), 518x518 (inference)

### Multi-Head Architecture

The model uses five specialized prediction heads:

#### 1. Segment Type Head
- **Architecture**: Linear(1024→512) → ReLU → Dropout(0.3) → Linear(512→3)
- **Output Classes**: Curve, Straight, Race_Start
- **Loss Function**: CrossEntropyLoss

#### 2. Curve Number Head
- **Architecture**: Linear(1024→512) → ReLU → Dropout(0.3) → Linear(512→14)
- **Output Range**: Curves 1-14
- **Loss Function**: CrossEntropyLoss

#### 3. Direction Head
- **Architecture**: Linear(1024→256) → ReLU → Dropout(0.3) → Linear(256→3)
- **Output Classes**: Left, Right, Unknown
- **Loss Function**: CrossEntropyLoss

#### 4. Racing Point Classifier
- **Architecture**: Linear(1024→512) → ReLU → Dropout(0.3) → Linear(512→4)
- **Output Classes**: Turn_in, Apex, Exit, None
- **Loss Function**: CrossEntropyLoss

#### 5. Coordinate Regressor
- **Architecture**: Linear(1024→512) → ReLU → Dropout(0.3) → Linear(512→2)
- **Output**: (x, y) normalized coordinates [0, 1]
- **Loss Function**: MSELoss (weighted 0.5)

### Total Loss
```
L = L_segment + L_curve + L_direction + L_point + 0.5 * L_coords
```

---

## Training Environment

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (tested on RTX 5090)
- 8GB+ GPU memory recommended

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd kart-coaching
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `torch` - PyTorch framework
- `transformers` - Hugging Face for DINOv2 model
- `pillow` - Image processing
- `scikit-learn` - Train/val splitting
- `numpy`, `scipy` - Numerical operations

### Dataset Preparation

The dataset follows a JSON annotation format with frame-level labels.

#### Annotation Format

Place your data in `data/annotations/default.json`:

```json
{
  "items": [
    {
      "attr": {"frame": 1},
      "image": {"path": "frame_001.jpg"},
      "annotations": [
        {
          "type": "label",
          "attributes": {
            "Type": "Curve",
            "Number": 1,
            "Direction": "Left"
          }
        },
        {
          "type": "points",
          "label_id": 1,
          "points": [0.45, 0.62]
        }
      ]
    }
  ]
}
```

#### Label Mappings

**Track Segment Types**:
- 0: Curve
- 1: Straight
- 2: Race_Start

**Racing Points**:
- 0: Turn_in
- 1: Apex
- 2: Exit
- 3: None (no point in frame)

**Directions**:
- 0: Left
- 1: Right
- 2: Unknown

#### Directory Structure
```
data/
├── annotations/
│   └── default.json
├── images/
│   └── default/
│       ├── frame_001.jpg
│       ├── frame_002.jpg
│       └── ...
```

### Dataset Creation

The `createDataset.py` module provides the `GoKartDataset` class:

```python
from transformers import AutoImageProcessor
from createDataset import GoKartDataset

processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
dataset = GoKartDataset("data/annotations/default.json", "data/annotations", processor)
```

**Features**:
- Automatic filtering of unlabeled frames
- Multi-point expansion (one sample per racing point)
- Normalized coordinate conversion
- Automatic image preprocessing

### Training

Run training with:

```bash
python training/training.py
```

**Training Configuration**:
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Weight Decay**: 0.01
- **Optimizer**: AdamW
- **Epochs**: 50
- **Train/Val Split**: 80/20
- **Workers**: 4 data loading threads

**Training Features**:
- Automatic checkpointing (saves best validation model)
- Real-time metrics: train loss, val loss, point accuracy, curve accuracy
- Early stopping based on validation loss
- Output saved to `model/best_model.pth`

**Monitoring**:
```
Epoch 1/50
 Train Loss: 2.3456 | Val Loss: 2.1234 | Point Acc: 65.32% | Curve Acc: 72.15%
```

### Model Export

After training, export the model for C++ inference:

```bash
python export_model.py
```

This creates `model/coach_model.pt` - a TorchScript traced model optimized for deployment.

**Export Process**:
1. Loads trained weights from `best_model.pth`
2. Traces model with dummy input (1, 3, 224, 224)
3. Saves TorchScript module
4. Validates output shapes

---

## Inference Environment

### Prerequisites

- CMake 3.18+
- CUDA Toolkit 12.0+
- LibTorch 2.0+
- OpenCV 4.x
- C++17 compiler (GCC 9+ or Clang)

### LibTorch Setup

1. Download LibTorch (cxx11 ABI version) from [pytorch.org](https://pytorch.org/get-started/locally/)

2. Extract to `/opt/libtorch`:
```bash
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.0+cu121.zip
sudo mv libtorch /opt/
```

3. Set environment variables:
```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH
```

### OpenCV Setup

Install OpenCV with CUDA support:

```bash
sudo apt update
sudo apt install libopencv-dev
```

Or build from source for better CUDA integration:
```bash
git clone https://github.com/opencv/opencv.git
cd opencv && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=7.5,8.6,8.9,9.0 \
      ..
make -j$(nproc)
sudo make install
```

### Building the Inference Engine

```bash
cd inference
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This produces the `inference` executable.

### Running Inference

Process a racing video:

```bash
./inference ../model/coach_model.pt input_video.mp4 output_video.mp4
```

**Arguments**:
1. Path to TorchScript model
2. Input video path
3. Output video path

### Inference Pipeline

#### 1. Frame Preprocessing
```cpp
cv::Mat → Resize(518x518) → BGR→RGB → Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]) → GPU Tensor
```

#### 2. Model Forward Pass
- Input: [1, 3, 518, 518] tensor
- Outputs: 5 tensors (segment, curve, direction, point, coords)

#### 3. Post-Processing
- Softmax for classification heads
- Argmax for class selection
- Coordinate denormalization
- Confidence extraction

#### 4. Visualization
- Circle overlay at predicted point (x, y)
- Color coding: Turn_in (green), Apex (red), Exit (blue)
- Text overlay: curve number, direction, point type, confidence

### Inference Output

The system annotates each frame with:
- Curve number and direction
- Racing point type
- Confidence score
- Visual marker at predicted location

Example overlay:
```
Curve 3 (Left) - Apex
Confidence: 92.5%
```

### Performance Considerations

**GPU Memory**:
- Model size: ~400MB
- Per-frame memory: ~50MB
- Recommended: 2GB+ VRAM

**Processing Speed**:
- RTX 3090: ~30 FPS (1080p video)
- RTX 4090: ~45 FPS (1080p video)
- Bottleneck: video I/O and encoding

**Optimization Tips**:
1. Use FP16 inference for 2x speedup
2. Batch processing for higher throughput
3. Hardware video decoding (NVDEC)
4. Asynchronous frame processing

### Code Structure

**`inference.h`** - Header file defining:
- `Prediction` struct (results container)
- `GoKartInference` class interface

**`inference.cpp`** - Implementation:
- Model loading and CUDA setup
- Frame preprocessing pipeline
- Prediction logic
- Video processing loop
- Visualization rendering

**`main.cpp`** - Entry point:
- Argument parsing
- Inference initialization
- Video processing orchestration

**`CMakeLists.txt`** - Build configuration:
- LibTorch linkage
- OpenCV integration
- CUDA architecture targeting

---

## Project Structure

```
kart-coaching/
├── model/
│   ├── coachModel.py           # PyTorch model definition
│   ├── best_model.pth          # Trained weights (PyTorch)
│   └── coach_model.pt          # Exported model (TorchScript)
├── training/
│   └── training.py             # Training script
├── inference/
│   ├── inference.h             # C++ header
│   ├── inference.cpp           # C++ implementation
│   ├── main.cpp                # Entry point
│   ├── CMakeLists.txt          # Build config
│   └── build/                  # Build artifacts
├── data/
│   ├── annotations/            # JSON labels
│   └── images/                 # Training frames
├── createDataset.py            # Dataset loader
├── export_model.py             # Model export script
└── requirements.txt            # Python dependencies
```

## Troubleshooting

### Training Issues

**CUDA Out of Memory**:
- Reduce batch size in `training/training.py`
- Use gradient accumulation
- Clear cache with `torch.cuda.empty_cache()`

**Poor Accuracy**:
- Increase training data
- Adjust loss weights
- Fine-tune learning rate
- Check data augmentation

### Inference Issues

**Model Loading Error**:
- Verify TorchScript export completed
- Check CUDA version compatibility
- Ensure model path is correct

**Slow Performance**:
- Enable TensorRT optimization
- Use smaller input resolution
- Check GPU utilization with `nvidia-smi`

**Build Errors**:
- Verify LibTorch path in CMakeLists.txt
- Check CUDA toolkit version
- Ensure OpenCV installation is complete

## Future Improvements

- [ ] Multi-scale feature fusion
- [ ] Temporal consistency (LSTM/Transformer)
- [ ] Real-time streaming inference
- [ ] Driver comparison mode
- [ ] Telemetry integration

