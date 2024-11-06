# Automatic Image Caption Generator

This project aims to build and compare different deep learning models to automatically generate captions for images. We explore three models for generating captions from images, using the Flickr8k dataset. The models are:

1. **Transformer Model**
2. **CNN + LSTM Model**
3. **CNN + LSTM with Attention Layer Model**

Each model's performance is evaluated and compared to determine the best approach for generating descriptive image captions.

## Project Overview

The goal of this project is to generate meaningful captions for images using deep learning techniques. The captions should describe the contents of the images accurately and in natural language. This project utilizes the Flickr8k dataset, which includes 8,000 images, each with multiple descriptive captions.

# Automatic Image Caption Generator

This project aims to build and compare different deep learning models to automatically generate captions for images. We explore three models for generating captions from images, using the Flickr8k dataset. Each model's performance is evaluated and compared to determine the best approach for generating descriptive image captions.

## Models

1. [**Transformer Model**](https://github.com/<username>/<repository-name>/wiki/Transformer-Model): This model uses a transformer-based architecture, known for handling sequential data and generating text. It leverages self-attention mechanisms to generate captions for images.

2. [**CNN + LSTM Model**](https://github.com/darship19/image-caption/wiki/Transformer-Model): A convolutional neural network (CNN) is used for image feature extraction, and a long short-term memory (LSTM) network is used for generating captions based on these features. This model has been popular for image captioning tasks.

3. [**CNN + LSTM with Attention Layer Model**](https://github.com/darship19/image-caption/wiki/cnn%E2%80%90lstm): This model extends the CNN + LSTM architecture by adding an attention mechanism, allowing the model to focus on different parts of the image while generating each word in the caption.



## Dataset

The **Flickr8k** dataset is used for training and evaluation. This dataset contains:
- 8,000 images, each with five different captions describing the image's content.
- Preprocessing includes resizing images, tokenizing captions, and splitting data into training, validation, and test sets.

## Implementation

### Requirements

- Python 3.x
- TensorFlow or PyTorch (depending on the framework used)
- Keras (if using TensorFlow)
- Numpy
- Matplotlib
- Pandas
- nltk (for text preprocessing)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/darship19/image-caption.git
   cd automatic-image-captioning
