# deep-learning-challenge

**UCSD Data Science Bootcamp Module 21 Challenge**

## Overview of the Analysis

The purpose of this analysis is to create a binary classifier that can predict whether applicants will be successful if funded by the nonprofit foundation Alphabet Soup and explain the effectiveness of the model. The data is from Alphabet Soup's business team and it contains more than 34,000 organizations that have received funding from Alphabet Soup over the years.

## Results

* Data Preprocessing:
  * The `IS_SUCCESSFUL` variable is the target or dependent variable for this model.
  * All variables excluding `IS_SUCCESSFUL`, `EIN`, and `NAME` are the features or independent variables for this model.
  * The `EIN` and `NAME` variables should be removed from the input data because they are neither targets nor features. These variables are more important as metadata for Alphabet Soup's business team and are not treated in the model.

* Compiling, Training, and Evaluating the Model:
  * For the first neural network, I selected 3 layers with 80 neurons in the first, 30 in the second, and 1 for the last layer and used the `relu` activation function for the first two layers and `sigmoid` activation function for the last layer. I chose 80 neurons for the first layer because there were many features to search through and the model has to account for a variety of circumstances in which a funding was successful. Then I chose 30 neurons mainly because I wanted the model to reduce in size and lead it to a sigmoid function in the output layer. I used the relu activation function because it seems to be a fairly common and widely used activation function for input and hidden layers. The output layer used a sigmoid activation function because this model is a binary classifier, and since this function outputs values between 0 and 1, it is ideal for this model.
  * Although I have attempted numerous runs of the model with various modifications, I was unable to achieve the target model performance of 75% accuracy.
  * In my attempts to increase model performance, I first utilized the `keras_tuner` module to run trials and identify the best hyperparameters. With this flexibility, I added the `leaky_relu` and `gelu` activation functions into the list of possibilities. I set the maximum number of neurons for the first and hidden layers to 500, so that the tuner has a wide range to search. I attempted to run the search with the number of layers set to a maximum of 50, but I quickly realized that the epoch runtime as well as the loss were relatively higher than when I tested with a maximum of 10 layers. Creating additional bins for the `APPLICATION_TYPE` and `CLASSIFICATION` variables resulted in less significant models with lower accuracy scores and higher loss scores.

## Summary

From the accuracy scores of the two binary classification models, we can conclude that both models do not achieve the target model performance of 75% and lack the confidence in their results. With the relatively high amount of time and effort that it took to optimize the model, the improved model is still insufficient in its ability to do binary classification from the processed data. The tuner did improve the accuracy and loss of the resulting model, with an increase in accuracy of 0.54% from the first model, but it still is below the goal of 75% at 73.62%. This model uses the `leaky_relu` activation function, has one input layer, three hidden layers, and one output layer, and the neuron counts all exceed 150 except for the output layer.

For a different approach to this problem, I recommend attempting the random forest model, as it is a supervised machine learning algorithm and can be helpful in determining feature importance to the model. Then we can remove or modify the relatively less important variables in the data to lessen the noise and inaccuracy in the resulting model.

## Code Sources and Locations

- Starter code - from `Starter_Code` folder

### AlphabetSoupCharity.ipynb

- Cell 8 {**loc** property} - from [Stack Overflow](https://stackoverflow.com/questions/38345213/using-value-counts-in-pandas-with-conditions)
- Cell 10 {**pd.get_dummies() and pd.concat()** functions and setup} - from `cc_preprocessing_solution.ipynb`
- Cells 14 to 17 {**Keras model** setup} - from `getting_real_solution.ipynb`

### AlphabetSoupCharity_Optimization.ipynb

- Cell 8 {**loc** property} - from [Stack Overflow](https://stackoverflow.com/questions/38345213/using-value-counts-in-pandas-with-conditions)
- Cell 10 {**pd.get_dummies() and pd.concat()** functions and setup} - from `cc_preprocessing_solution.ipynb`
- Cells 14 to 19 {**Keras tuner and model** setup} - from `tune_up_solution.ipynb`
- Cell 14 {**Layer activation functions**} - from [Keras Doc](https://keras.io/api/layers/activations/#creating-custom-activations)
- Cell 15 {**Hyperband directory** parameter} - from [Stack Overflow](https://stackoverflow.com/questions/59439124/keras-tuner-search-function-throws-failed-to-create-a-newwriteablefile-error)