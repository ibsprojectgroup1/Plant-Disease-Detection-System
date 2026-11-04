
# MODEL TRAINING AND EVALUATION
## TECHNICAL HANDOVER DOCUMENT

### 1. Project Title

Development of a Deep Learning-Based Model for Cassava and Maize
Leaf Disease Detection and Classification Using Image Processing Techniques.

### 2. Department Responsibility

Model Training and Evaluation Department.

The department was responsible for training, comparing, selecting,
and evaluating deep learning models for the classification of
cassava and maize leaf images.

### 3. Dataset Information

Dataset Version: 2

Total images: 4,000
Training images: 2,560
Validation images: 726
Testing images: 714

Image dimensions: 224 x 224 pixels
Number of classes: 10

### 4. Trained Models

Primary model: ResNet50
Model file: trained_models/ResNet50_V2_Final.keras
Model size: 200.94 MB

Alternative model: EfficientNetB0
Model file: trained_models/EfficientNetB0_V2_Final.keras
Model size: 16.30 MB

### 5. Model Evaluation Results

| Metric | ResNet50 | EfficientNetB0 |
|---|---:|---:|
| Test Accuracy | 85.43% | 82.35% |
| Macro Precision | 85.91% | 83.47% |
| Macro Recall | 85.81% | 82.98% |
| Macro F1 | 85.50% | 82.21% |

ResNet50 was selected as the primary model because it achieved
higher overall test performance. EfficientNetB0 is retained as
a smaller alternative.

### 6. Class Mapping

0 - cassava_cbb
1 - cassava_cbsd
2 - cassava_cgm
3 - cassava_cmd
4 - cassava_healthy
5 - maize_common_rust
6 - maize_gray_leaf_spot
7 - maize_healthy
8 - maize_northern_leaf_blight
9 - maize_not_maize_leaf

The class order must remain unchanged when interpreting model outputs.

### 7. Preprocessing

Both models use 224 x 224 RGB images.

The models have different preprocessing requirements.
The deployment team must preserve the exact preprocessing
pipeline used during training for each model.

The ResNet50 model contains a custom ResNet50Preprocessing layer.
Its implementation is supplied in preprocessing_code.py.

The EfficientNetB0 input pipeline must be verified against
the original training notebook and saved model before deployment.

### 8. Prediction Script

The initial integration script is supplied as prediction.py.

It demonstrates model loading, image input, class selection,
and probability output.

The script must be validated against the original training
pipeline before production use.

### 9. Evaluation Evidence

The evaluation folder contains:

- Classification reports
- Confusion matrices
- Evaluation metrics
- Test predictions
- Test probabilities

The model_comparison.csv file contains the comparative results.

### 10. Known Limitations

The models do not achieve perfect classification performance.
Cassava disease categories, particularly cassava_cbb, remain
challenging and may be confused with other cassava classes.

The system should communicate predictions as model estimates,
not confirmed agricultural diagnoses.

The maize_not_maize_leaf class should not be treated as a
universal rejection mechanism for every possible non-leaf image.

### 11. Responsibilities of the Pipeline and Deployment Department

The receiving department should:

1. Load and validate the selected trained model.
2. Implement the correct model-specific image preprocessing.
3. Integrate the ten-class mapping.
4. Build the image input and prediction pipeline.
5. Display predicted class and confidence score.
6. Test the system using independent leaf images.
7. Handle invalid images and unsupported inputs.
8. Integrate the model into the project's application or interface.
9. Document deployment tests and system limitations.

### 12. Handover Status

The two trained models have been saved.
Both models passed model-loading and output-shape verification.

The training and evaluation evidence has been organised
for transfer to the Pipeline and Deployment Department.

The prediction script is an initial integration implementation
and requires further validation before production deployment.
