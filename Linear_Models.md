## Linear Model

Our parametric model has two inputs. It takes in an image, and a set of parameters. It should will spit out 10 numbers, for each class (still using our CIFAR-10 example)

<img src='https://snag.gy/JWUAmv.jpg' />

- 32 height in pixels
- 32 width in pixels
- 3 color channels

## What is f(x,W)? 

f will be the algorithm or model, 


## Linear version of f(x,W): just multiply:

<img src='https://snag.gy/dBWhfT.jpg' />

The image matrix here is rolled out to a long vector `32 x 32 x 3=3072`. The target output is 10 class scores, so the weight matrix will be `3072 x 10`. There will also be a bias term `b` will offset for different classes in the data. Below is a simple example if the images were `2x2`

<img src='https://snag.gy/3y1MW4.jpg' />

### What can we learn from the weights?

The weight matrix can be reformed into `32x32x31 images and be considered "templates" that images are compared to.

<img src='https://snag.gy/9t3WOh.jpg' />

Note that the linear model only has 1 template per class. If there's very different images of a cat, with different backgrounds or different colors, the linear template is trying to average out all the different pictures.

<img src='https://snag.gy/Hkmpyr.jpg' />

Thinking about this spatially, in a high-dimensional space, the linear classifier is trying to find a plane (linear separator) that will divide the class from the others.

<img src='https://snag.gy/eZp2CX.jpg' />