# Adversarial Images improves Image Recognition
Adversarial examples are commonly viewed as a threat to ConvNets. Here an opposite perspective is investigated such that adversarial examples can be used to improve image recognition. We investigate two adversarial training schemes such as **AdvProp**, **NoisyStud**. 

## Testing
We used ImageNet-A (https://www.kaggle.com/paultimothymooney/natural-adversarial-examples-imageneta) as our dataset which is composed of nearly 7500 images.
We used baseline preprocessing, AdvProp plus AutoAugment and noisyStudent plus RandAugment with B3 and B4 weighting of the EfficientNet (i.e., B3-4 checkpoints are used.). 


We used **filtering.py** file to filter invalid Jpeg images from dataset which results crash of code execution. 
**Main.py** is main file to test models with ImageNet-A.

## Checkpoints
Use the following link to download checkpoint(s):

https://drive.google.com/drive/folders/1PiMxZbYvAsnXVEWltDfcXlbb1yM4wJ8a?usp=sharing
