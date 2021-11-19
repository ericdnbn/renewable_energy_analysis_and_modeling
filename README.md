# Solar Energy Analysis And Modeling
### Author: Eric Denbin

In this project I am going to employ data analysis, machine learning, and deep learning to assess and model the cost of solar energy in the US. 

### Author: Eric Denbin

<p align="center">
  <img src=images/derma.jpeg />
</p>

## Business Understanding

Skin cancer is the most common form of cancer in the United States and worldwide. In the United States, more people are diagnosed with skin cancer each year than all other forms of cancer combined.

<p align="center">
  <img src='images/skin_cancer_infographic.png' width=500 height=200 />
</p>

Skin lesions are typically first diagnosed using clinical methods, such as single image expert consensus or serial imaging of dermoscopic images. At this stage of the diagnostic process, medical professionals are visually examining the size, color, shape, uniformity, and location of skin lesions. 

<p align="center">
  <img src='images/single_image_consensus_example.png' width=550 height=250 />
</p>

If a diagnosis is uncertain, more clinical tests may be performed. These include blood tests, allergy tests, and skin swabs, among others. At this point, if a lesion is suspicious for malignancy, or the diagnosis is still uncertain, the specific type of lesion is determined by analyzing a biopsy under a microscope.

<p align="center">
  <img src='images/biopsy.jpeg' width=400 height=200 />
</p>

When it comes to clinically diagnosing skin lesions, medical professionals often misdiagnose benign lesions as being suspicious for malignancy. According to a study published in April of 2021 in the Dermatology Research And Practice journal, while 99.06% of the lesions clinically diagnosed as being benign were actually benign, just 82.85% of all benign lesions were identified (https://www.hindawi.com/journals/drp/2021/6618990). This results in an inefficient use of resources. A study published in the Journal Of Clinical Oncology in June of 2018, estimated that biopsies of benign lesions cost between $624 million and $1.7 billion over the course of a year (https://ascopubs.org/doi/abs/10.1200/JCO.2018.36.15_suppl.e18903). Given those facts, a model could be used to identify misdiagnosed benign lesions, and therefore reduce the number of biopsies taken of benign lesions.



## Data Understanding

My dataset consists of 7,179 dermoscopic images of skin lesions from the International Skin Imaging Collaboration(ISIC) archive (https://www.isic-archive.com/). All patients were 10-90 years old and the images were taken in the course of clinical care.The following file structure provides the ground truth labeling needed to train the models. If you wish to run my code, you will need to download images from the ISIC archive into the same directory format:
```
└── dermoscopic_images
    ├── train
    │    ├──benign
    |    ├──malignant
    │    └──unknown
    └── test
         ├──benign
         ├──malignant
         └──unknown
```

<p align="center">
  <img src='images/class_examples.png' width=550 height=150 />
</p>

The ISIC archive contains over 150,000 images, 70,000 of which have been made public. The images can be downloaded from the gallery, and are labeled by  I downloaded only dermoscopic images to ensure a certain standard of quality across the dataset. The archive contains 23,704 dermoscopic images of benign lesions, 2,240 dermoscopic images of malignant lesions, and 2,212 dermoscopic images of unknown lesions. I downloaded 2,401 images of benign lesions for training and validation, and 980 images of benign lesions for testing. I downloaded 1500 dermoscopic images of malignant lesions for training and validation, and 600 for testing. I downloaded 1500 dermoscopic images of unknown lesions for training and validation, and 600 for testing. The class balance in my training set is 44.44% images of benign lesions, 27.78% images of malignant lesions, and 27.78% images of unknown lesions.

<p align="center">
  <img src='images/skin_lesion_class_balance.png' width=450 height=350 />
</p>

A significant limitation of the data is that the vast majority of patients represented in the archive have fair skin. This presents an ethical issue, given that it introduces bias to the model that could make it less effective in making predicitons about particular groups of patients. Unfortunately, the ISIC does not provide any information about the demographic breakdown of the archive. The only information I could find about demographics as it relates to the ISIC archive came from an article published in 2018 in the Atlantic written by Ashley Lashbrook that addresses the issue of bias in AI as it relates to making predictions about skin lesions, which said, "The ISIC, too, is looking to expand its archive to include as many skin types as possible." Without any other information, however, it is difficult to say if the archive has expanded and made any progress to date. Given this imbalance, it is important to keep in mind that the results of this project do not indicate how the model would perform broadly.



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


