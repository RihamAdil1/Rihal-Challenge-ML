
# N24News Project 

# Rihal-Challenge-ML
This challenge for Rihal.om
https://github.com/rihal-om/rihal-codestacker/tree/main/ML

## Introduction
In today's digital age, the abundance of news articles available online has led to an overwhelming amount of information for readers to navigate. With such a flood of content, finding relevant news has become increasingly challenging. Consequently, there is a growing demand for effective news classification systems, which are vital tools for organizing and accessing important content with ease.

Furthermore, there is an urgent need for systems capable of captioning images, providing clear and concise descriptions that aid in understanding and engagement. Additionally, there is an increasing desire for systems that can generate abstracts from the main body of articles, condensing the key points into brief summaries for easy consumption and understanding.



## Description

This project aims to solve the challenges faced by Rihal Company which it consists of 5 research questions:
### RQ1: Develop a model that can categorize news articles into their respective categories.
### RQ2: Generate abstracts that provide a clear and concise summary of the article.
### RQ3: Generate captions for each news article's image that accurately reflect the content.
### RQ4: Implement a real-time UI web app for inference where it allows the user to upload an article body, image, and a title, and then return its category, a caption, and an abstract (you could use tools such as Streamlit or Gradio).
### RQ5:Detect if the news article is related to Palestine and categorize it under a new subcategory called "FreePalestine".

## Notebooks
[`notebook1.ipynb`](path_to_notebook1.ipynb): This notebook contains data preprocessing steps.


1. `notebook1.ipynb`: This notebook contains data preprocessing steps.
2. [`notebook2.ipynb`]([path_to_notebook1.ipynb](https://colab.research.google.com/drive/1_CUOKjuVvRKgrmlJEI2wVn-L9bon8s4b#scrollTo=Bq42J4w6CIuS)): This notebook contains code of generating abstracts from Article body
3. 
4. `notebook3.ipynb`: This notebook showcases results and analysis.
   ----------------------------------------------------------------------------------------------------------------------------------------------
   
## Approaches:

## <small>Approach 2: Generate abstracts that provide a clear and concise summary of the article.</small>

  ### <small>1.Data Preprocessing.</small>
* Extract the article bodies and corresponding abstracts from the dataset.
* Apply tokenization to convert the text data into sequences of tokens.
* Fit the Tokenizer on both articles and abstracts to build the vocabulary and convert words to indices.

   ### <small>2.Model Architecture.</small>
* The encoder consists of an embedding layer followed by an LSTM layer.
* The decoder also consists of an embedding layer and an LSTM layer.
* An attention mechanism is incorporated to help the decoder focus on relevant parts of the input sequence during decoding.
* The model predicts the next token in the sequence using a softmax activation function.

  ### <small>3.optimization and loss function.</small>
* The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function.

  ### <small>4.Training stage.</small>
* The model should be  trained in a loop over 10 epochs.
* Data will be shuffled, and batches of input-output pairs will be fed  into the model for training.
* The encoder and decoder input sequences are trimmed or padded to ensure consistent length.
* The train_on_batch method will be used to train the model on each batch.

  ### <small>5.Validation stage.</small>
* A separate validation dataset will be used to evaluate the model's performance during training.
* After each epoch, the model's performance on the validation dataset will be assessed using the evaluate method to compute the loss.

  ### <small>6.Adjustments and Tuning:.</small>
 * batch size, sequence length, and the number of epochs will be adjusted
 * Hyperparameter tuning and experimentation until get the optimal model

  

  ## <small>Approach 3: Generate captions for each news article's image that accurately reflect the content.</small>
  ### <small>1.Data Preprocessing.</small>
* For text(headline,Abstract and Caption will be extracted and tekonized).
* For images, images will be extracted,resized and normalized and then will undergoe to ResNet Model to extract the important features.

   ### <small>2.Model Architecture.</small>
* Input Layers will be Defined( headline, abstract, and image features).
* embedding layers to encode headline and abstract inputs.
* using LSTM units to encode headline and abstract sequences.
* Concatenate LSTM outputs with image features.
* attention mechanism will be applied  to focus on relevant features.
* Decoding attention output sequences.
* a dense layer with softmax activation for caption generation will be performed.
* Define model with inputs and output.
*Compiling  the model with optimizer and loss function.

  ### <small>3.Model Training.</small>
*the number of epochs and batch size for training will be setermined and adjusted as needed.
* fitting training data into model.
* Evaluating the model on the validation data during the training.
* Monitoring the training progress and performance on the validation set.

  ### <small>4.Model Evaluation.</small>
* Generating captions for the testing set.
* Preparing reference and generated captions.
* Compute BLEU score (evaluation metric).



  

## Setup

### Prerequisites

Before running this project, ensure that you have Python installed on your system.
these notebooks has been runned on google colab 
To run this project locally, make sure you have Python installed. Then, install the required dependencies using:

```bash
pip install -r requirements.txt

