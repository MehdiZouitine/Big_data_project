
# Zouitine Mehdi, Sekhar Ali, Lasserre Aymeric, Tarati Manutea

Necessary package : 
* numpy
* torch
* transformers
* pyspark
* sklearn
* matplotlib
* seaborn
* mlflow
* pandas
* tqdm
# Introduction
In this project the goal is to predict the job of a person based on a small text description. 
The data are one dataset containing: the sex, the id, and the text associated to each person. Each row correponds to one specific person.
The other dataset contains the labels associated to the different jobs (it is the target). 
We have each dataset in two different versions : one for the train set the other for the test set.

# Data preparation

The goal of this part is to go from a text to a matrix containing only numbers in order to use it later to do multiclasses classification. To do so we are going to use the number of
times a word appears in the text. But to apply this transformation we will need to gather all the different form of the word together. 


## Text cleaning
The first cleaning step is to remove all the punctuation signs (becasue they are useless for us). To do so we use regex to keep only letters from the text.
Then we tokenize each one of our documents using a tokeniser. This means that we create for each document a list and that each element of this list correponds to a word of the document.
basically we do a split on the space of the document.
Once it is done we remove the stopwords. They are words such as "but", "the" or "by". 
They are just noise which means that they do not bring any useful information in the sentence and they will just complexify the computation later on.
Then we go on to the final step of the cleaning process : the stemming. This step allow to bring back every word to its root-word. For example fishing, fished and fisher will be associated
to fish. We use a famous python library called nltk and its snowball stemmer. 
At the end we have a column vector containing in each line a list of words that is unique and representative of the text description of the individual.


## TF transformation
Now that the text is ready to be transformed into float with are going the use TF transformation. TF stands for Term Frequency and it is the simple idea to assocaite to each term in the 
document the number of time it appears. 
After applying the transformation we receive for each document (row) a vector containing :
	-the total number of different word into the document and then 
	- a list of float that correponds to the listof words previsoulsy created.
It is on this vector that the machine learning part is going to take place

# A Machine Learning Model - Logistic Regression Classifier

In order to use our word vectorization to predict the job of a person, we will use a state of the art machine learning model which is the Logistic Regression.
It is a first insight on how good we are doing to predict the jobs categories.

On PySpark, we use the LogisticRegression function from the pyspark.ml.classification module.

We will use the default parameters for a matter of time.

After fitting our Logistic Regression Classifier on the training set, we then evaluate it on the test set by computing the f1-score. 

As results, we obtain a score of 0.712. 

We will now see how to build a better model based on Neural Networks.

# A Deep Learning Model - Bert transformer + fine tuning

To improve our score we used fine tuning on model bert.
By using bert model we improved our score in the test (from **0.712** to **0.809**).

We used many tricks to improve our results, that are documented directly in the code.
Some examples :
* Focal loss
* Hard negative mining
* Different learning rate and weight decay

We also try thing that didn't work well : 
* Undersampling and Oversampling (Because the dataset is highly unbalanced)
* Use bert large
* Use distil bert
* Use more dense layer for the fine tuning head

We also used mlflow to monitor our model and generate many confusion matrix to analyze the weak point of our model
