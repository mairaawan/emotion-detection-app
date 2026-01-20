📄 Project Abstract

Facial Emotion Recognition (FER) is an important area of computer vision and human–computer interaction. This project presents a deep learning–based system for recognizing human emotions from facial images using a Convolutional Neural Network (CNN) implemented in PyTorch.

The proposed system classifies facial expressions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. Images are preprocessed using grayscale conversion, resizing, normalization, and extensive data augmentation techniques to improve robustness and generalization.

To address overfitting and class imbalance, the model incorporates Batch Normalization, Dropout, Class Weights, Label Smoothing, Learning Rate Scheduling, and Early Stopping. Experimental results show that while the training accuracy exceeds 90%, the test accuracy improves significantly to approximately 62–70%, demonstrating effective generalization.

The trained model is saved for evaluation and deployment. The system is designed to be easily extendable to real-time emotion detection using a webcam and a Streamlit-based interactive web interface.
