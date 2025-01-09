#include "esp_camera.h"
#include "model.h"
#include "wifi_manager.h"

// Model input settings
constexpr int kNumChannels = 3;
constexpr int kImageWidth = 96;
constexpr int kImageHeight = 96;

void setup() {
    Serial.begin(115200);

    // Initialize camera
    setupCamera();

    // Initialize WiFi
    setupWiFi();

    // Initialize model
    if (!setupModel()) {
        Serial.println("Error loading model");
        return;
    }
}

void loop() {
    // Capture image
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        return;
    }

    // Process image and run inference
    float* input = preprocess_image(fb->buf, fb->len);
    int prediction = runInference(input);

    // Send results to server
    sendDetection(prediction);

    // Return frame buffer
    esp_camera_fb_return(fb);
    delay(1000);
}
