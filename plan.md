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

## Next Steps

1. Collect real-world test data
2. Fine-tune model for specific bird species
3. Implement power management
4. Add data logging capabilities
5. Create user interface for configuration
