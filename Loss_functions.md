# Loss Functions and Optimization

### Quantifying the "badness" of a model
Depending on the type of data the "badness" of the model is calculated through a **loss function**. This is a loss function. The simplest example of something like this is predicting housing prices:

<img src='https://snag.gy/BSJWAU.jpg' style='width:400px' />

For this dataset, the loss function is mean squared error of house price:

$$ L(y_{pred}, y_{true}) = (y_{pred} - y_{true})^2  $$

But in classification terms, what is our loss function? When predicting prices, guessing 1,000,000 sale price vs. the true 200,000 is very wrong. But in classification, how "wrong" is guessing 'cat' instead of 'dog'?

### Multiclass SVM Loss

