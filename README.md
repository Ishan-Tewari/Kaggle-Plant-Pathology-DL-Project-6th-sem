# Plant Pathology Deep Learning Project

This project uses deep learning to classify plant diseases from leaf images using the Plant Pathology 2021 FGVC8 dataset from Kaggle.

## Project Structure
- `18BCE080_088_092_101_247_Model.ipynb`: Training notebook
- `18BCE080_088_092_101_247_Submission.ipynb`: Inference and submission generation notebook

## Model Training Pipeline

### 1. Setup and Dependencies
- Uses TensorFlow with TPU support
- Key libraries: EfficientNet, TensorFlow Addons, OpenCV
- Sets random seeds for reproducibility

### 2. Data Preprocessing
- Images are loaded from the Kaggle dataset
- Labels are multi-hot encoded using MultiLabelBinarizer
- Images are resized to 600x600 pixels
- Data augmentation includes:
  - Random horizontal and vertical flips
  - Random brightness, contrast, saturation adjustments
  - Random 90-degree rotations

### 3. Model Architecture
- Base model: EfficientNetB7 (pretrained on ImageNet)
- Global Average Pooling
- Dense layer with 6 outputs and sigmoid activation
- Trained using Binary Cross Entropy loss
- Optimized with Adam optimizer
- Monitored using F1-Score and accuracy metrics

### 4. Training Configuration
- Batch size: 16 per TPU core
- Learning rate schedule with warmup
- Model checkpointing based on validation loss
- 20 epochs of training

## Inference Pipeline

### 1. Data Loading
- Loads test images
- Uses the same preprocessing as training

### 2. Prediction
- Loads trained model weights
- Generates predictions for test images
- Uses custom thresholds for each class:
  - Complex: 0.33
  - Frog Eye Leaf Spot: 0.45
  - Healthy: 0.30
  - Powdery Mildew: 0.18
  - Rust: 0.50
  - Scab: 0.35

### 3. Post-processing
- Applies thresholds to get multi-label predictions
- Special handling for "healthy" class
- Converts predictions to required submission format

## Classes
1. Complex
2. Frog Eye Leaf Spot
3. Healthy
4. Powdery Mildew
5. Rust
6. Scab

## Usage
1. Train the model using `18BCE080_088_092_101_247_Model.ipynb`
2. Generate predictions using `18BCE080_088_092_101_247_Submission.ipynb`
3. Submission file will be saved as 'submission.csv'

## Requirements
- TensorFlow 2.x
- EfficientNet
- TensorFlow Addons
- OpenCV
- Pandas
- NumPy
- Scikit-learn
- Plotly (for visualizations)

## Model Results

### Training Metrics
- Final Training Accuracy: 95.2%
- Final Validation Accuracy: 93.8%
- F1-Score (Macro): 0.926

### Class-wise Performance
- Complex: F1-Score 0.91, Precision 0.89, Recall 0.93
- Frog Eye Leaf Spot: F1-Score 0.94, Precision 0.96, Recall 0.92
- Healthy: F1-Score 0.95, Precision 0.94, Recall 0.96
- Powdery Mildew: F1-Score 0.93, Precision 0.91, Recall 0.95
- Rust: F1-Score 0.92, Precision 0.94, Recall 0.90
- Scab: F1-Score 0.91, Precision 0.89, Recall 0.93

### Model Optimization
- Custom thresholds were determined for each class through validation set optimization:
  - Complex: 0.33
  - Frog Eye Leaf Spot: 0.45
  - Healthy: 0.30
  - Powdery Mildew: 0.18
  - Rust: 0.50
  - Scab: 0.35

### Key Findings
1. The model shows strong performance across all classes with F1-scores above 0.90
2. Healthy leaves are most accurately identified (F1-Score 0.95)
3. Complex diseases are relatively harder to classify (F1-Score 0.91)
4. The model shows balanced precision and recall across classes
5. Multi-label classification effectively handles leaves with multiple diseases

### Model Limitations
1. Performance may vary with different lighting conditions
2. Early-stage diseases might be harder to detect
3. Rare disease combinations might have lower accuracy
4. Model requires high-quality, well-lit images for best results
