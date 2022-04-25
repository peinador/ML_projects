# ML_projects

This repository currently contains three short machine learning projects in python jupyter notebooks. 
The ML models where implemented using scikit-learn and keras.
The projects are:
- Iris_dataset: classifyier models for the iris flower dataset. 
- Fashion-MNIST: Convolutional Neural Network classifyer for the Fashion MNIST data set.
- LSTM_trade: Long Short-Term Memory model to predict 10-day return of HSBCA stocks.

## Iris_dataset

This notebook implements different ML models to classify iris flowers into three categories (spieces). The dataset contains 150 samples and 4 features. The implemneted models are:
- SVM
- K-Nearst Neighbours 
- Neural Net

All three models achieve accuracies greater than 0.9. 
Additionally, a PCA was applied to reduce the dimensionality to 3, obtaining better separated clusters.

![PCA of the iris datset](/Iris_dataset/PCA.png)

## Fashion-MNIST

The Fashion-MNIST datset is a benchmark dataset consisting of 70K 28x28 greyscale images of clothing items. 
The items are labeled into 9 categories. 

The selected model is a CNN with 7 hidden layers, containing convolutional, max pooling and dense layers.
The model accuracy on the validation set is slighly above 90%. This is a satisfactory result considering that the low resolution of the sample images makes the classification of some of the images difficult for humans. For example, some of the wrong predictions were:

![Wrong predictions of the CNN](/Fashion-MNIST/WrongPred.png)

The last part of the notebook uses the [Grad-CAM++](https://arxiv.org/pdf/1710.11063.pdf) method to iterpret the model.
The result is a heatmap that represents the most significant regions of the images for the CNN classification.

![Grad-CAM++ example](/Fashion-MNIST/Gradcam.png)

# LSTM_trade

The goal of this porject was to create a ML model to predict the 10-return of the HSBCA stocks, given the open/closing price  and volume of a previus time interval.
The raw data, obtained from Yahoo Finance, is processed into different financial indicators using the TA-lib library. A total of 5 signals are selected as input for the model.
The selected model was a LSTM network, given its natural hability to capture correlations and model time series. 
The model contains three LSTM layers, two dropout layers and two dense layers (including output layer).
The time window is 10 trading days.
The predictions sucessfully follows the trends both in the training and testing set. The following figure shows the real return vs prediction for a random segment of the training and test series.

![Train and test performace](LSTM_trade/traintest.png)

A common pitfall of this type of strock prediction models is that sometimes the model learns to repeat the previous day point, yielding a price curve offset by one day. 
If one plots the curves over a long time domain, both curves overlap giving the impression of a very robust prediction.
Our model however, outperforms the silly copy-the-previous-day strategy. The following plot shows how the model performace improves the silly model both for training and test sets during training.

![Train and test performace](LSTM_trade/MSE.png)
