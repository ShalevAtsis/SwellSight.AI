# SwellSight Project Overview

## 1. Motivating Use Case

### Use Case Background
Surfers rely on low-quality beach-camera images to decide when to surf conditions. These images provide visual information but lack objective measurements.

### Problem
* **Subjectivity:** Current wave assessment is based on guesswork and varies across surfers and locations
* **Lack of Data:** There is no tool that translates a 2D image into quantifiable physical metrics (like exact height in meters or breaking type)

### Why the Problem Is Challenging
* Waves change constantly with lighting, angle, wind, and camera quality
* Key attributes like breaking type and surfability are hard to judge from a single image
* No existing labeled dataset contains detailed wave attributes

### Idea
Develop a Computer Vision model that automatically analyzes beach images to extract objective physical parameters (Wave Height, Breaking Type), providing a standardized basis for conditions.

---

## 2. Project Definition

**Given a single image of ocean waves, use a Multi-Task Deep Learning model to extract objective physical wave parameters.**

### Input
* One RGB image from a beach camera or a surfer's photograph

### Output
* **Wave height:** continuous value in meters (Regression)
* **Wave type and breaking style:** A-frame, closeout, beach break, point break (Classification)
* **Wave Direction:** left, right, both (Classification)

---

## 3. Project Novelty

* **Overcoming the lack of labeled surf data** by generating a custom dataset using Synthetic Depth Maps
* This technique allows complete control over wave geometry to train the model with perfect ground-truth labels
* **First model to shift from subjective "quality estimation" to the precise extraction of physical attributes** from 2D images
* A unified Deep Learning pipeline that simultaneously solves regression (height) and classification (wave type) tasks for complex fluid dynamics

---

## 4. Models and Methods

### Processing Pipeline
* **Synthetic Data Generation:** We create a labeled dataset by generating Depth Maps (representing wave geometry) and converting them into photorealistic images using ControlNet and Stable Diffusion
* **Model Inference:** The trained Deep Learning model processes a single 2D image to extract physical wave attributes

### Models & Techniques
* **Data Generation:** Using Depth Maps as geometric conditions allows us to create synthetic waves with known, ground-truth physical parameters (Height, Shape)
* **Wave Analysis Model:** We utilize a Multi-Task Learning architecture with shared feature extraction and three specific output heads

### Adjustments & Fine-Tuning
* A pre-trained image encoder is fine-tuned on our custom synthetic dataset. The shared backbone learns robust features relevant for all three tasks simultaneously (geometry, orientation, and scale)
* **Domain adaptation:** To bridge the gap between synthetic training data and real beach photos, we apply:
  * Aggressive data augmentation (lighting, noise, perspective)
  * Validation against a small set of real-world reference images

---

## 5. Data Specification and Generation

### Data Requirements
* **Training Set:** A large-scale, fully synthetic dataset generated with precise geometric control
* **Validation Set:** A small collection of real-world beach images, manually labeled, used solely for Domain Adaptation and final testing

### Labeling Strategy
* **Automatic Ground Truth:** Since we define the wave's geometry (e.g., "Height = 1.8m") before generating the image, the labels are inherently 100% accurate. No manual labeling is required for the training set

### Synthetic Data Generation
* **Geometry First:** We programmatically generate Depth Maps that define the exact 3D structure of the wave (Height, Shape, Angle)
* **Texture Rendering:** These Depth Maps serve as strict conditions for ControlNet (integrated with Stable Diffusion) to generate photorealistic water textures and lighting

---

## 6. Metrics and KPIs

### Measuring Results
Performance is evaluated on the model's ability to accurately extract physical attributes from the images.

### Quality Assessment per Task
* **Wave Height:** 
  * Metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) in meters
* **Wave/Breaking Type:** 
  * Metrics: Accuracy, F1-Score, and Confusion Matrix
* **Wave Direction:** 
  * Metrics: Accuracy and F1-Score

### Evaluation Protocol
* **Synthetic Validation:** Testing on a held-out set of synthetic images to verify learning capacity across all three heads
* **Real-World Validation:** Running the model on real beach images to assess domain adaptation success and visual consistency

---

## 7. Technical Architecture

### Core Model Design
**Backbone encoder**
* A pre-trained image encoder (e.g., ResNet/ConvNeXt/ViT) serving as a **shared feature extractor**
* Encoders are **pre-trained on large image datasets** and then fine-tuned on surf images (synthetic + real)

**Task heads (multi-task architecture)**
From the shared feature vector `z`:
1. Height regression head → scalar (meters)
2. Wave type head → 4-way softmax
3. Direction head → 3-way softmax

Each head is a small MLP / CNN block on top of `z`.

**Loss function**
Multi-task loss combining:
* L1 / L2 for height
* Cross-entropy for type and direction
with per-task weights to balance optimization.

### Output Schema Example
```json
{
  "wave_height_m": 1.8,
  "wave_type": "A_FRAME",
  "direction": "RIGHT",
  "probs": {
    "wave_type": {"A_FRAME": 0.73, "CLOSEOUT": 0.10, "BEACH_BREAK": 0.09, "POINT_BREAK": 0.08},
    "direction": {"LEFT": 0.12, "RIGHT": 0.68, "BOTH": 0.20}
  }
}
```

---

## 8. Summary

**SwellSight** is a **multi-task, GenAI-powered vision model** that:

* Takes a single beach-cam image as input
* Outputs height, type, and direction as objective physical parameters
* Is trained primarily on synthetic diffusion-generated waves (with exact labels) and validated on a smaller real dataset
* Uses a shared, fine-tuned vision encoder with separate heads for each wave attribute
* Evaluates performance with MAE/RMSE for regression and accuracy/F1 for classification tasks
* Represents the first system to extract precise physical wave parameters from 2D images rather than subjective quality assessments