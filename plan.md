# Smart Birdfeeder with TinyML Bird Species Classification

## Project Structure
📁 smart-birdfeeder/
├── 📁 models/
│ ├── 📁 training/
│ │ ├── pre-process.py
│ │ ├── train.py
│ │ └── evaluate.py
│ └── 📁 converted/
│ └── bird_model.tflite
├── 📁 esp32-firmware/
│ ├── 📁 src/
│ │ ├── main.cpp
│ │ ├── camera.h
│ │ ├── model.h
│ │ └── wifi_manager.h
│ ├── platformio.ini
│ └── README.md
├── 📁 webapp/
│ ├── 📁 frontend/
│ └── 📁 backend/
└── README.md

## Progress Tracking (Updated 2024)

### Completed Milestones ✅
1. Dataset Infrastructure
   - Dataset loading and validation implemented
   - Split ratios optimized: Train (94.2%), Validation (2.9%), Test (2.9%)
   - Cache implementation with parallel processing
   - Image preprocessing pipeline established
   - Tensor conversion and batching system working

2. Model Architecture Selection
   - EfficientNet-B2 chosen for optimal accuracy/size trade-off
   - Model configuration validated for ESP32-S3 constraints
   - Preprocessing parameters defined (see preprocessor_config.json)

### Current Status 🔄
1. Training Pipeline
   - Data loading: ✅ Complete
   - Preprocessing: ✅ Complete
   - Model training: ⏳ In Progress
   - Model optimization: 🔜 Pending
   - Dataset shuffling optimization: 📝 TODO

2. System Architecture
   - Dataset Protocol implementation: ✅ Complete
   - TensorFlow pipeline: ✅ Complete
   - ESP32 integration: 🔜 Pending

### Technical Specifications 📊
1. Dataset Metrics
   - Training: 84,635 images (2,645 batches)
   - Validation: 2,625 images (83 batches)
   - Test: 2,625 images (83 batches)
   - Image dimensions: 96x96x3 (RGB)
   - Memory footprint per batch: ~884KB
   - Current state: Sequential ordering (to be optimized)

2. Performance Benchmarks
   - Processing speed: ~1,700 images/second
   - Memory utilization: ~2GB peak
   - Batch processing: 32 images/batch

## Deep Learning Fundamentals

### Neural Network Architecture
1. **EfficientNet-B2 Design**
   - Compound scaling method
   - Depth: 1.2x base network
   - Width: 1.1x base network
   - Resolution: 260x260 pixels
   - Parameters: ~7.8M

2. **Transfer Learning Strategy**
   - ImageNet pre-training
   - Fine-tuning approach:
     * Gradual unfreezing
     * Progressive layer adaptation
     * Learning rate scheduling

3. **Optimization Pipeline**
   - Batch normalization (eps=0.001)
   - Dropout rate: 0.3
   - Hidden activation: Swish
   - Loss function: Cross-entropy

## 1. Understanding Dataset Preparation

### What are Dataset Splits?

In machine learning, we split our data into different sets:

- Training set (70-80%): Used to teach the model
- Validation set (10-15%): Used to tune hyperparameters
- Test set (10-15%): Used for final evaluation

The preprocessing code is already set up in:

```python
models/training/pre-process.py
```

## 2. Model Training and Optimization

### Initial Training

We're using EfficientNet-B2 as shown in:

```python:models/bird-species-classifier/train.py
import torch
from datasets import load_dataset
import evaluate
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification, TrainingArguments, Trainer

# ... existing code ...
```

### Optimizing for ESP32-S3

Add the following code to convert and optimize the model:

```python
import tensorflow as tf
def optimize_for_esp32(model_path, output_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    # Enable optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Enable quantization
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Representative dataset for quantization
    def representative_dataset():
        for data in train_ds.take(100):
            yield [data[0].numpy()]

    converter.representative_dataset = representative_dataset

    # Convert model
    tflite_model = converter.convert()

    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
```

## 3. ESP32-S3 Implementation

### PlatformIO Configuration

```ini
[env:esp32s3]
platform = espressif32
board = esp32s3-devkitc-1
framework = arduino
monitor_speed = 115200
board_build.flash_mode = qio
board_build.f_cpu = 240000000L
build_flags =
    -DCORE_DEBUG_LEVEL=5
    -DBOARD_HAS_PSRAM
    -mfix-esp32-psram-cache-issue
lib_deps =
    esp32-camera
    TensorFlowLite_ESP32
    ArduinoJson
    ESP Async WebServer
```

### Camera Setup

```cpp
#include "esp_camera.h"
#include "camera_pins.h"

void setupCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_RGB565;
    config.frame_size = FRAMESIZE_96X96;
    config.jpeg_quality = 12;
    config.fb_count = 1;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x", err);
        return;
    }
}
```

## 4. Inference Pipeline

```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

class BirdDetector {
private:
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input_tensor;

public:
    void processImage(camera_fb_t* fb) {
        // Preprocess image
        uint8_t* rgb_buffer = preprocess_image(fb);

        // Run inference
        memcpy(input_tensor->data.int8, rgb_buffer, 96*96*3);
        interpreter->Invoke();

        // Get results
        TfLiteTensor* output = interpreter->output(0);
        int8_t max_score = -128;
        int max_index = -1;

        // Find highest confidence prediction
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (output->data.int8[i] > max_score) {
                max_score = output->data.int8[i];
                max_index = i;
            }
        }

        // Send notification if confidence is high enough
        if (max_score > CONFIDENCE_THRESHOLD) {
            sendNotification(getBirdName(max_index));
        }
    }
};
```

## 5. Notification System

```python
# Backend API (FastAPI)
from fastapi import FastAPI
from twilio.rest import Client

app = FastAPI()

@app.post("/notify")
async def send_notification(bird_data: BirdDetection):
    # Send SMS via Twilio
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    message = client.messages.create(
        body=f"A {bird_data.species} was spotted at your feeder!",
        from_=TWILIO_PHONE_NUMBER,
        to=USER_PHONE_NUMBER
    )

    # Store detection in database
    await store_detection(bird_data)
    return {"status": "notification sent"}
```

## Machine Learning Concepts Explained

### 1. Model Quantization

Quantization reduces model size by converting floating-point weights to integers. This is crucial for ESP32-S3 deployment as it:

- Reduces memory usage
- Speeds up inference
- Maintains reasonable accuracy

### 2. Transfer Learning

We're using EfficientNet-B2 as our base model because:

- It's pre-trained on ImageNet
- Has excellent accuracy/size trade-off
- Well-suited for mobile deployment

### 3. Inference Optimization

Key optimization techniques used:

- Int8 quantization
- PSRAM usage for larger buffers
- Batch processing avoidance
- Memory-efficient image preprocessing

### 4. Real-time Processing

Tips for real-time bird detection:

- Use motion detection to trigger inference
- Implement frame skipping
- Buffer management for continuous operation
- Power management considerations

## Memory Management and Optimization

### 1. Tensor Operations
1. **Data Flow Architecture**
   ```
   Raw Image (RGB) → Preprocessed Tensor → Model Input
   [H×W×3, uint8] → [96×96×3, float32] → [1×96×96×3, int8]
   ```

2. **Memory Optimization Techniques**
   - Batch processing (32 images/batch)
   - In-place operations where possible
   - Efficient tensor allocation
   - Memory-mapped file handling

### 2. ESP32-S3 Constraints
- PSRAM: 8MB available
- Flash: 16MB maximum
- Processing: 240MHz dual-core
- Memory alignment: 32-bit

## Model Training Pipeline

### 1. Dataset Preparation (✅ Completed)
```python
# Current preprocessing flow:
Raw Image → ImageData → ProcessedBatch → TensorFlow Dataset
```

### 2. Training Configuration
```python
training_args = TrainingArguments(
    finetuned_model_name,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01
)
```

### 3. Quantization Strategy
1. **Post-Training Quantization**
   - Int8 quantization
   - Representative dataset: 100 samples
   - Dynamic range optimization

2. **Model Size Reduction**
   - Original: ~30MB
   - Post-quantization target: <4MB
   - Accuracy impact: <5%

## Implementation Roadmap

### Phase 1: Model Training (Current)
- [x] Dataset preprocessing
- [x] Training pipeline setup
- [ ] Initial model training
- [ ] Validation metrics
- [ ] Hyperparameter tuning

### Phase 2: Optimization
- [ ] Int8 quantization
- [ ] Model pruning
- [ ] Performance benchmarking
- [ ] Memory profiling

### Phase 3: ESP32 Integration
- [ ] Camera interface
- [ ] Model loading
- [ ] Inference pipeline
- [ ] Power optimization

## Technical Considerations

### 1. Training Infrastructure
- GPU Requirements: CUDA-compatible
- Memory: 16GB RAM minimum
- Storage: 100GB for dataset/checkpoints
- Framework versions:
  * TensorFlow: 2.x
  * PyTorch: 2.x
  * Python: 3.8+

### 2. ESP32 Development
```ini
[env:esp32s3]
platform = espressif32
board = esp32s3-devkitc-1
framework = arduino
monitor_speed = 115200
board_build.flash_mode = qio
board_build.f_cpu = 240000000L
```

### 3. Performance Metrics
1. **Training Metrics**
   - Batch processing: ~1,700 images/second
   - Training time: ~10 hours estimated
   - Validation frequency: Every epoch

2. **Inference Metrics (Targets)**
   - Latency: <500ms
   - FPS: >2
   - Power consumption: <250mW

## Risk Mitigation

### 1. Technical Risks
- Memory constraints on ESP32
- Real-time performance
- Power consumption
- Temperature management

### 2. Mitigation Strategies
1. **Memory Management**
   - Streaming inference
   - Buffer optimization
   - Dynamic memory allocation

2. **Performance Optimization**
   - Motion-triggered inference
   - Batch size optimization
   - Pipeline parallelization

## Next Steps

1. **Immediate (24-48 hours)**
   - Complete model training
   - Initial performance evaluation
   - Quantization testing

2. **Short-term (1-2 weeks)**
   - ESP32 integration
   - Power optimization
   - Initial field testing

3. **Long-term (1-2 months)**
   - Web interface development
   - Data collection system
   - Long-term testing

### TODOs for Optimization 📝
1. **Dataset Shuffling**
   - Implement tf.data.Dataset shuffling
   - Add buffer size optimization
   - Verify random distribution
   - Benchmark impact on training
