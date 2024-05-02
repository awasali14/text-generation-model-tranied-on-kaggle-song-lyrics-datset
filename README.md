# Text-generation-model-tranied-on-kaggle-song-lyrics-datset


## Overview
This repository contains a TensorFlow-based implementation of a text generation model designed to create song lyrics. The model leverages deep learning techniques, specifically Long Short-Term Memory (LSTM) networks, to predict sequences of words that form coherent and creative lyrics.

## Project Description
The project was created to explore and demonstrate the capabilities of sequence models, particularly LSTM and Bidirectional LSTM networks, in generating text. It addresses the challenge of producing text that is not only grammatically correct but also contextually meaningful.

## Problem Solved
This implementation tackles the complexity of language modeling, which involves understanding the dependencies between words in a sequence to generate text that follows a logical and stylistic structure. It provides a framework for experimenting with different network architectures to enhance the quality of generated text.

## Learnings
The project highlights several key aspects of natural language processing, including:

Preprocessing text data for neural network training.
Utilizing Tokenizer and padding techniques to convert text data into sequences that can be fed into an LSTM model.
Implementing a Bidirectional LSTM to improve the context awareness of the model.
Training a neural network on a sequence data to predict the next word in a sequence.
Generating new text based on a seed to evaluate the model's language generation capabilities.


## Setup Dependencies
TensorFlow
NumPy
Pandas
Matplotlib

## Data
The dataset consists of 250 song lyrics fetched from an online source, which is preprocessed to remove punctuation, convert to lowercase, and tokenize into sequences.

## Model Architecture
The model uses:

Embedding Layer: To create dense vector representations of words.
Bidirectional LSTM Layer: To capture dependencies from both previous and upcoming words for better context understanding.
Dense Layer: With softmax activation to predict the probability distribution of the next word.


## Training
The model is trained using categorical crossentropy loss and the Adam optimizer, with metrics set to monitor accuracy. The training involves using callbacks such as Early Stopping to prevent overfitting and Model Checkpoint to save the best model based on validation accuracy.

## Results
Model performance is visualized in terms of training accuracy. Post-training, the model is used to generate lyrics starting from a given seed text, demonstrating its ability to compose new lyrics that are influenced by the learned text patterns.

