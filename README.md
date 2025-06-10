# Transformer-Based Video Classification in PyTorch using Rotary Positional Embeddings

## Description

This project demonstrates the implementation of a transformer-based video classifier in PyTorch.

High-level features from the first 24 frames of each video were extracted with a pre-trained image classification model. 
A transformer-based classification model was then trained on the sequences of extracted frame features to recognize the action performed in a video. 

The model consists of a TransformerEncoder module with Rotary Positional Embedding (RoPE) and 4 attention heads, 
followed by a classifier module which performs global average pooling over the time dimension before generating predictions.

## Dataset and Augmentation

The model was trained on a subsampled version of the UCF101 dataset containing only the 5 most frequent classes. This subsampled dataset was created by Paul (2021), 
and can be downloaded from [here](https://github.com/sayakpaul/Action-Recognition-in-TensorFlow/releases/download/v1.0.0/ucf101_top5.tar.gz).
Data augmentation was performed by producing a grayscale copy of each training sample, doubling the size of the training dataset.

## Model Performance

The model achieved a maximum validation accuracy of 99.1 %.

## Credits

Paul, S. (2023). *Video Classification with Transformers*. https://keras.io/examples/vision/video_transformers/

Sarkar, A. (2025). *Transformer Model Tutorial in PyTorch: From Theory to Code*. https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch



