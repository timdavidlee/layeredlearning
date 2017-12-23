# Intro to Image Classification

Ideally, image identification would be a simple function as listed down below:

```python
def classify_image(image):
	# make some magic
	return what_is_it?
```

## Previous Attempts: Complicated Rules

Previously its been attempted to develop a high-level approach for digesting and understanding images. 
<img src='https://snag.gy/Nn7BMi.jpg' style='width:600px'/>

For example, image edges and corners are important for image recognition, how about creating a specific set of rules for recognizing specific sets of edges as a human and another set as as a gopher, etc? This turns out not to be a good approach as the rules are very brittle. And if the classifier's goal changes mid-stream, from recognizing cats to recognizing trucks, the work has to start all over again. **This approach is not sustainable or scalable for all the varieties of objects in the world.**

## The data driven approach

#### 1. Collect a dataset of images and Labels 
Note: this is a huge amount of work to look at all these images, standardize the size and label them. There's a lot of good datasets out there

<img src='https://snag.gy/4rlpZw.jpg' style='width:500px'/>


#### 2. 'Train' as classifier

```python
def train_model(images, labels):
	# machine learning!
	return model
```
The model will take in all the data, and spits out a model that will summarize how to detect all these images. 

#### 3. Evaluate the classifier on New Images

```python
def predict(model, new_images):
	# use model to predict labels
	return test_labels
```

Finally this 'trained' model will be used on new images that will be able to recognize cats and dogs.

### Challenges to Image Recognition:
<img src='https://snag.gy/ba1rUp.jpg' />




## A simple example: Nearest Neighbor
Now the image recognition task has been broken into two different phases, but these two phases are not exclusive to image recognition but is used in machine learning and deep learning at large. To get used to the paradigm, let's consider a simple classifier: **nearest neighbor**.

1. `train` - will memorize all the images and all the labels

2. `predict` - will predict the label based on the most similar training image

### A Simple dataset CIFAR-10 (see dataset page)

<img src='https://snag.gy/M38Xb9.jpg' style='width:700px' />

In this approach, test images were compared to the entire CIFAR-10 database, and our classifier compared the absolute visual differences. On the right side, the test image is shown and all similar images are noted by the arrow. The resulting clustering of images that were "most similar" had the most visually similiar, but were not always correct. Consider the 4th row of the test results, similar images all have a white background with a colored centered object. Sometimes tits a frog, but sometimes its a fighter jet. 



### The python code:

```python
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred

```

### Train Function: Memorize the data

Note that we memorize the training data pixel by pixel for every single image (X)

```
  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y
```
### Predict: find the closest image

As we compare the test image to every single pixel in the training set. Find the closest image and return that label as our "best guess"

```
for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
```
 
### Speed Considerations

- **Train speed:** O(1)
- **Predict speed:** O(N) - compared against every possible image

This is quite slow, if the training set has millions of images, prediction will take a very long time. This is also reverse of the ideal. Ideally training is time and work intensive prediction runtime is fast, so it can be run in the browser or mobile device.

### What does nearest neighbors look like?

### k-Nearest Neighbors?

Instead of looking the single closest neighbor, an alternative method is the k-nearest neighbor. This looks at the k-nearest points and does a majority vote. This can either be pure voting or weighted on distance.

<img src='https://snag.gy/stKpRj.jpg'/>

## Broader Question: How to compare Images?

### Manhatten Distance (L1)

<img src='https://snag.gy/B30wDA.jpg' />

<img src='https://snag.gy/9g4Yyk.jpg' />

Choosing different distant metrics make assumptions about the space. Between these two metrics, L1 depends on the coordinate system. L2 stays the same with coordinate transformations.

By using different distant metrics, this kNN paradigm can be used for a number of different applications. For text, all that needs to be understood is a distance function between texts or words. 

<img src='https://snag.gy/DeMk0h.jpg' />




