# MiniGPT From Scratch

This directory contains a simple implementation of a character-level Generative Pre-trained Transformer (GPT) model, built from scratch using PyTorch. The purpose of this project is to understand the core mechanics of GPT models in a minimalist setting.

## Project Structure

- **`train.py`**: The main script that handles data loading, model definition, training loop, and text generation.
- **`input.txt`**: The source text file used for training the model. The model learns to generate text based on the patterns found in this file.
- **`requirements.txt`**: Lists the Python dependencies required to run the code.

## How it Works

1.  **Data Preparation**: The script reads text from `input.txt` and creates a character-level vocabulary. It maps each unique character to an integer index.
2.  **Model Architecture**: A simple neural network (`MiniGPT`) is defined with:
    -   An embedding layer that converts character indices into dense vectors.
    -   A linear layer that projects the embeddings back to the vocabulary size to predict the next character.
3.  **Training**: The model is trained using the Adam optimizer and Cross Entropy Loss. It learns to predict the next character in a sequence given a context window.
4.  **Generation**: After training, the model generates new text by sampling from the predicted probability distribution of the next character, iteratively building a sequence.

## Usage

1.  **Install Dependencies**:
    Ensure you have Python installed, then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    Add your training text to `input.txt`. The model will learn from whatever text you provide here.

3.  **Run Training**:
    Execute the training script:
    ```bash
    python train.py
    ```
    The script will print the loss during training and output a generated text sample upon completion.

## Requirements

-   Python 3.x
-   PyTorch
-   NumPy
