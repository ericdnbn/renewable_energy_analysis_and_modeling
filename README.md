# Solar Energy Analysis And Modeling
### Author: Eric Denbin

In this project I am going to employ data analysis, machine learning, and deep learning to assess and model the cost of solar energy in the US. 



## Business Understanding



## Data Understanding



## Data Analysis



## Modeling With Neural Networks

The first simple model consists of a fully connected dense neural network with two hidden layers, plus an output layer. This model serves as a proof of concept and provides baseline metrics.

The following is the confusion matrix it produced:

<p align="center">
  <img src='images/confusion_matrix_fsm.png' width=600 height=375 />
</p>

The first simple model returned a validation accuracy of 44.44%, as it predicted every image to be part of the benign class. Given that I trained it for just five epochs as a proof of concept, these results were as much as I expected.


To improve on the first simple model, I began iterating on convolutional neural networks. The following are various adjustments made over these iterations to improve model performance:
 - Adding more dense layers
 - Adding convolutional layers
 - Adding dropout layers
 - Adding batch normalization layers
 - Using L2 regularization
 - Trying different kernel sizes

Using convolutional neural networks, the validation accuracy of the models increased along with precision and recall in predicting the benign class. However, the validation accuracy reached a ceiling around 68% and the model's precision and recall in predicting the benign class was volatile.

In an effort to improve model performance, and particularly precision and recall when predicting the benign class, I began exploring transfer learning. I used the pre-trained VGG16 model with the 'imagenet' weights as a base, and the same architecture from the best convolutional neural network to construct the fully connected dense layers. The following are other adjustments I made as I continued iterating:
 - Increasing the number of dense layers 
 - Increasing the number of nodes in the first hidden layer with each additional layer

 
Collectively, I iterated through more than ten models, going from a fully connected dense neural network, to convolutional neural networks with custom architecture, and finally, to convolutional neural networks with the pre-trained VGG16 model as a base, and custom architecture for the fully connected dense layers. My final model has the following architecture:

<p align="center">
  <img src='images/final_model_summary.png' width=560 height=600 />
</p>

Below is a diagram of the best model that depicts the portion of the VGG16 model I used as a base, as well as the fully connected dense layers:

<p align="center">
  <img src='images/VGG16_visual.png' width=700 height=300 />
</p>

<p align="center">
  <img src='images/net2vis_layers.png' width=700 height=275 />
</p>



## Final Model Evaluation

I trained the best model for 25 epochs with a batch size of 128 images. The following shows the confusion matrix results after evaluating the best model on the testing dataset:

<p align="center">
  <img src='images/confusion_matrix_final_model.png' width=563 height=375 />
</p>

The accuracy of the best model on the test set was just 75.31%, but in terms of my key metric, F1 score as it relates to predicting the benign class, it performed quite well.

Out of 946 lesions the model predicted to be benign, 933 are benign, 3 are malignant, and 10 are unknown. This means that its precision when predicting lesions to be benign is 98.63%, which is less than a half of a percent lower than the precision of medical professionals clinically diagnosing lesions as being benign. 

Out of 980 total benign lesions, the model was able to identify 933. This means that its recall when predicting lesions to be benign is 95.20%, which is about 13% better than the recall of medical professionals clinically diagnosing lesions as being benign. 

Using these values for precision and recall, the F1 score as it relates to predicting the benign class comes out to 96.88. 

It is important to note that the model does have trouble distinguishing between malignant and unknown lesions, as demonstrated by the 447 malignant lesions the model predicted to be unknown. However, as it relates to the business problem, this is irrelevant, because a biopsy would be taken regardless of whether a lesion is predicted to be malignant or unknown. What is important is that there are no benign lesions predicted to be malignant, and only 4.80% of all benign lesions are predicted to be unknown.

Given the model’s precision and recall as it relates to predicting the benign class, it could successfully be used to identify misdiagnosed benign lesions, and therefore reduce the number of biopsies taken of benign lesions.



## Exploring The Blackbox




## Conclusions

### Recommendations




### Next Steps




## For More Information


