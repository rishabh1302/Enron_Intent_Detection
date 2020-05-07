# Enron_Intent_Detection
Intent detection on Enron email set. We define "intent" here to correspond primarily to the categories "request" and "propose". In some cases, we also apply the positive label to some sentences from the "commit" category if they contain datetime, which makes them useful. Detecting the presence of intent in email is useful in many applications, e.g., machine mediation between human and email. The dataset contains parsed sentences from the email along with their intent (either 'yes' or 'no'). You need to build a learning model which detects whether a given sentence has intent or not. Its a 2-class classification problem.

# Results :

79.5% ACCURACY with TF-IDF and RandomForests after parameter tuning.

# Inference & Interpretation:


Random Forests performed better than LGBM and XGBoost, maybe because of the below reason : 

The model tuning in Random Forest is much easier than in case of XGBoost. In RF we have two main parameters: number of features to be selected at each node and number of decision trees. RF are harder to overfit than XGBoost. I did not try tuning for XGBoost as it is extremely compute intensive and time consuming.

It is generally the case that a well-tuned GBM can work better than a RF. However, the GridSearchCV recommendation was to also use a square root max feature parameter, which worked well in the case of random forests!


# Approach to Solve - Rishabh

We have two sets here - train and test set.

### EDA:
We perform an EDA of the train set to check the following :
1) Character/String Lengths of both labels (YES and NO)
2) Word Counts of both labels (YES and NO)
3) Top unigrams and bigrams after removing stopwords from the train set.
4) TSNE graph to check distinguish-ability of the labels.
5) Calculating the unique word counts in the train set.


### Model Building - 7 Classifiers

After getting to know the data a little better after the EDA, these insights are used to start constructing a  binary text classification model.

Steps:
1) [Data Preprocessing] 
Word contractions are expanded such as transforming -   ["couldn't've" to "could not have"], ["didn't" to "did not"] and ["doesn't" to "does not]. Data is then cleaned to remove stopwords, stripping spaces, special characters, HTML tags, Email IDs & related tags.
The text is now lemmatized where the words are transformed into their root form.

2) [Data Transformation]
Training text data is transformed using TF-IDF to create word vectors, the feature names and their idf score is also calculated to better understand the vectorization part. The train and test shapes are displayed. The target variable in the form of Yes and No is transformed into 1's and 0's respectively.

#### Training Data is split into 80% Train & 20% Validations.

3) [Data Modelling]

The training data is then went onto fit upon the below 7 classifiers:
a) LogisticRegression
b) SVC
c) MultinomialNB
d) KNeighborsClassifier 
e) RandomForestClassifier
f) LGBMClassifier
g) XGBClassifier

The validation set is then predicted for to check the classifier performance, majority of the classifiers give 72% validation accuracy.

4) [Test Set Prediction]

All the above models were then used to predict for the test set, the best accuracy was given by ### Random Forests at 77.7%.

5) [LSTM + CNN Model]

Another model is built using a keras deep learning architecture to perform binary text classification. However this model only achieves 72% test accuracy, probably because of lack of data fed into such a system.

5) [Hyper-Parameter Tuning with GridSearchCV for TF-IDF + Random Forests]

Since random forests were performing well, we tried to tune the hyper-parameters using grid search and obtain the best ones. These parameters will be used to retrain a model and check the test accuracy.

6) [Applying Tuned Params to Predict Test Set]

The tuned params which were the output of the GridSearchCV are now used to predict for the test set. The model now achieves 
**79.5% accuracy**






