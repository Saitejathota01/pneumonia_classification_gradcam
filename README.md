

---

# Pneumonia Classification using Grad-CAM

## Overview

This Jupyter Notebook demonstrates the process of classifying chest X-ray images to detect pneumonia using Convolutional Neural Networks (CNNs). Additionally, it employs Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize the areas of the X-ray images that contribute most to the model's decision-making process.

## Dataset

The dataset used in this notebook is sourced from Kaggle and can be found [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). It contains chest X-ray images categorized into three classes:
1. **NORMAL**: No signs of pneumonia.
2. **PNEUMONIA**: Signs of bacterial pneumonia.
3. **VIRAL PNEUMONIA**: Signs of viral pneumonia.

## Notebook Contents

### 1. **Importing Libraries**

   The notebook begins by importing essential libraries such as TensorFlow, Keras, NumPy, Pandas, Matplotlib, and others required for data preprocessing, model building, training, and visualization.

### 2. **Data Loading and Preprocessing**

   - **Loading Data**: The dataset is loaded from the specified directory structure.
   - **Data Augmentation**: To enhance the model's generalization capabilities, data augmentation techniques such as rescaling, zooming, shearing, rotation, and horizontal flipping are applied.
   - **Data Splitting**: The dataset is split into training, validation, and testing sets.

### 3. **Model Building**

   - **CNN Architecture**: The model is built using a Convolutional Neural Network (CNN) with multiple layers including Convolution, MaxPooling, Dropout, and Dense layers.
   - **Compilation**: The model is compiled with appropriate loss functions and optimizers.

### 4. **Model Training**

   The model is trained on the training set, and its performance is validated on the validation set. The training process includes monitoring of accuracy and loss.

### 5. **Evaluation**

   - The model is evaluated on the test set to determine its final performance.
   - Metrics such as accuracy, precision, recall, and F1-score are calculated.

### 6. **Grad-CAM Visualization**

   - **Grad-CAM Implementation**: Grad-CAM is implemented to visualize the important regions in the X-ray images that influence the model's predictions.
   - **Visualization**: The notebook demonstrates how to overlay the Grad-CAM heatmap on the original images for better interpretability.

### 7. **Conclusion**

   A summary of the findings and potential future work is discussed.

## How to Run the Notebook

1. **Prerequisites**: Ensure you have Python and Jupyter Notebook installed.
   
2. **Dataset**: Download the dataset from Kaggle and place it in the appropriate directory as expected by the notebook.

3. **Running the Notebook**: Open the notebook in Jupyter and execute the cells sequentially.

## Results

- The model achieves significant accuracy in detecting pneumonia and differentiates between bacterial and viral pneumonia effectively.
- Grad-CAM visualizations help in understanding the model's decision-making process, providing insights into the areas of the chest X-rays that are crucial for classification.

## References

- Dataset: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Grad-CAM: Selvaraju, R.R., Cogswell, M., Das, A. et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. Int J Comput Vis 128, 336–359 (2020).

---
      
