# Logistic-Regression-with-Handwritten-Images
Use a logistic regression model to classify digits from a data set of handwritten digits. We will consider classification between two different digits at a time (this is simpler, and will help us try to understand how logistic regression works better).We will initialize a logistic regression model, and learn its parameters from the images and the labels, using the call to “fit”. Our logistic regression model has a weight vector w composed of weights wi, one for each pixel
xi. We also have a bias b; these weights are learned from a dataset.Once we have learned the weights, our logistic regression model predicts the probability that a new input image x has a positive label or a negative label.

We'll start off classification between 1 and 7.

## Some images that were labeled as a digit seven in the training set were misclassified as being a digit one by the classifier. Find the image of a seven that had the highest probability of being a one (include a picture). Explain why the digit was classified as a one instead of as a seven (include a visualization, like we did above).
Some images that were labeled as a digit seven in the training set were misclassified as being a digit one by the classifier. Find the image of a seven that had the highest probability of being a one (include a picture). Explain why the digit was classified as a one instead of as a seven (include a visualization, like we did above).

## Some images that were labeled as a digit one in the training set were misclassified as being a digit seven by the classifier. Find the image of a one that had the highest probability of being a seven (include a picture). Explain why the digit was classified as a seven instead of as a one (include a visualization, like we did above).
This digit was classified as a seven instead of a one because the relevant weights used to calculate its probability for each digit strongly favored seven (over 70%) even though it was labeled one. This can be seen by the second image because more red pixels (positive-7) fall within the digits area.

## Train another logistic regression model using any other pair of digits i and j, besides 1 and 7. Provide a single example of a digit labeled i being misclassified as j, with a visualization of why it was misclassified (like we did above). (Note that this is only possible if some image of a digit was misclassified, i.e., if it did not get 100% accuracy.)
When training 0 vs 6, an image of the zero digit was wrongfully classified as 6. This can be seen in the second image where the positive class’ (6) weights (red) make up most of the zero image which explains the over 90% probability of it being a 6 according to the classifier.
