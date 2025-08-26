# 🐾 Pet Emotion Recognition using Few-Shot Learning
A deep learning project 



### 📌 Overview
A deep learning project to classify *pet emotions (cats & dogs)* like happy, sad, angry, relaxed from facial images.  
Instead of relying on huge datasets, this project applies *Few-Shot Learning with Prototypical Networks* so that the model can recognize emotions even with limited training samples.

### 🔧 Tech Stack
- *Python*  
- *PyTorch & Torchvision* (deep learning & pretrained models)  
- *Scikit-learn* (evaluation metrics)  
- *NumPy, tqdm* (data handling & progress tracking)  
- *Matplotlib/Seaborn* (visualization of results)  

### 📊 Workflow
1. *Dataset*: Pet facial expression images (Kaggle).  
2. *Preprocessing*: Resizing, normalization, and data augmentation.  
3. *Model: Few-Shot Learning approach using **Prototypical Networks* with a ResNet34 backbone.  
4. *Evaluation*: Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.  

### 🚀 Features
- Works well even with small datasets (few-shot learning).  
- Uses *embedding-based prototypes* instead of softmax classification.  
- Evaluates performance across multiple test episodes for robustness.  
- Detailed classification report + confusion matrix.  
