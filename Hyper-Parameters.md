# Hyper Parameter Tuning
< insert images >

**Idea 1:** Choosing the hyper parameters that work best on data. 

BAD: will memorize the data, and offers no guarantee on newer unseen data

**Idea 2:** Split data into train and test, choose hyper parameters that work best on test data. 

BAD: same issue as before, 

**Idea 3:** Split data into three different sets. Train, validation, and finally test set. 


The goal of machine learning is to work with unseen data (coming from the wild)


### Cross Validation

Mainly used in small datasets. 

**Idea 4:** Split data into folds, try each fold as validation nad average the results

Not used in deep learning since the training is very computationaly expensive, this practice won't be used as much. 

### Is it possible that the Test set does not represent the training set?

The assumption is that the test set is IID. The typical approach is to collect all the data all at once and then partition it after the fact. This is also the reason to becareful of collecting data over time, to ensure that nothing underlying has changed. 