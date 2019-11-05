# LDA_HAN-for-News-Biased-Detection

This is Keras implementation of the Hierarchical Network with Attention architecture [(Yang et al, 2016)](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf).
Instead of using standard word embedding alone, this work applys a topic-aware word embedding which combines word vectors with transposed topic-word distribution and such distribution is a global weighting of the dimensions of the word-topic vector. Once the model got the document representation, it also combines with the document-topic distribution before feeding to softmax at the final prediction.

## Experiments
All results are calculated by the mean of 10-fold cross validation five times on the [Hyperpartisan News Detection](https://pan.webis.de/semeval19/semeval19-web/) by-article training set data (645 news articles in total).  


| Model | Accuracy |
| --- | --- |
| Transformer | 72.12% | 
| LDA-Transformer | 71.56% | 
| Kim-CNN | 72.95% | 
| LDA-Kim-CNN | **73.47%** | 
| RNN-Attention | 73.63% |
| LDA-RNN-Attention | **73.75%** | 
| ESRC| 71.81% |
| LDA-ESRC | **73.69%** |
| HAN | 75.69% |
| LDA-HAN | **76.52%** | 

## Preparation / Requirements

* Python 3.6 (Anaconda will work best)
* Tensorflow version 1.13.0
* Keras version 2.2.4
* Gensim 3.8.0
* Spacy version 2.1.16
* flask 1.1.1

Preparation steps:
1. `mkdir checkpoints data embeddings history lda_stuff` for store trained models, training data set, glove & lda embeddings, training logs, gensim models respectively.
2. Convert original data xml file to tsv format. (see [this](https://github.com/GateNLP/semeval2019-hyperpartisan-bertha-von-suttner/tree/4b1d74b73247a06ed79e8e7af30923ce6828574a) for how it works).
3. `python ./utils/lda_gen.py -train True -H True -T 300` for generating lda topic embedding with 300 dimensions.
4. `python main.py -train True` for training the model.
5. `python visulization.py` for building the web application and visualize attention distribution and predictions.

## Example 1:
![alt text](https://github.com/yjiang123/LDA_HAN-for-News-Biased-Detection/blob/master/images/Discursive.png "Discursive News")
## Example 2:
![alt text](https://github.com/yjiang123/LDA_HAN-for-News-Biased-Detection/blob/master/images/Tendentious.png "Tendentious News")
