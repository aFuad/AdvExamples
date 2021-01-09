# Adversarial Images improves Image Recognition
Adversarial examples are commonly viewed as a threat to ConvNets. Here we present an opposite perspective: adversarial examples can be used to improve image recognition. We propose an enhanced adversarial training scheme **AdvProp**.

## Testing
We used ImageNet-A[https://www.kaggle.com/paultimothymooney/natural-adversarial-examples-imageneta] as our dataset which is composed of nearly 7500 images.
We used Baseline preprocessing, AdvProp and NoisyStudent with B3 and B4 checkpoints of the EfficientNet. 


We used **filtering.py** file to filtering Invalid Jpeg images from dataset which results crash of the execution of the code. 
**Main.py** is main file to test models with ImageNet-A.

