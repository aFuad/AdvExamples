#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
from scipy import misc
import numpy as np 
import matplotlib.pylab as plt
import argparse
import math
import matplotlib.pyplot as plt
import os
from os import listdir
from PIL import Image 



name = '/home/ubuntu/.jupyter/MyNotebooks/imagenet-a/imagenet-a/n01498041/0.000348_chameleon _ box turtle_0.55540705.jpg'
path3 = '/home/ubuntu/.jupyter/MyNotebooks/imagenet-a/imagenet-a/'
dirs=os.listdir(path3)

count=0
# Traversing directories 
for i in range(0, len(dirs)):
  dirs2=os.listdir(path3+dirs[i])
  for y in range(0, len(dirs2)):
    mainpath=path3+dirs[i]+'/'+dirs2[y]
  

# 1000-ImagenetA Labels 
d = {}
with open("/home/ubuntu/.jupyter/MyNotebooks/ImageNet-A_labels/imagenet1000_clsidx_to_labels.txt") as f:
    for line in f:
       (key, val) = line.split(': ')
       d[int(key)] = val[1:-3]

#print(d)
# print(d[1][0])

# Imagenet-A Folder Labels 
folders = {}
with open("/home/ubuntu/.jupyter/MyNotebooks/ImageNet-A_labels/map_clsloc.txt") as f:
    for line in f:
      a=line.split(' ')
      folders[a[0]] = a[2][:-1]

#print(folders)


randdirs1=[]
with open("/home/ubuntu/.jupyter/MyNotebooks/RandomDirectories0.txt") as f:
    for line in f:
       val = line.split('\n')
       randdirs1.append(val[0])
    
print(randdirs1)        

randdirs2=[]
with open("/home/ubuntu/.jupyter/MyNotebooks/RandomDirectories1.txt") as f:
    for line in f:
       val = line.split('\n')
       randdirs2.append(val[0])
        

randdirs3=[]
with open("/home/ubuntu/.jupyter/MyNotebooks/RandomDirectories2.txt") as f:
    for line in f:
       val = line.split('\n')
       randdirs3.append(val[0])


# In[29]:


path3 = '/home/ubuntu/.jupyter/MyNotebooks/imagenet-a/imagenet-a/'
dirs=os.listdir(path3)
from random import seed
from random import randint
seed(1)

selected_img = [0] * 7500
#print(selected_img)

directories=[]
count=0
# Traversing directories 
for i in range(0, len(dirs)):
  dirs2=os.listdir(path3+dirs[i])
  for y in range(0, len(dirs2)):
    directories.append(path3+dirs[i]+'/'+dirs2[y])
    count+=1

#print(directories[10])   

for a in range(0, 3):
    for i in range(0, 150):
        value = randint(0, len(directories) - 1)
        while(selected_img[value]!=0):
            value = randint(0, len(directories) - 1)
        a_file = open("RandomDirectories"+str(a)+".txt", "a")
        selected_img[value]=1
        a_file.write(str(directories[value]) + "\n")
        print(directories[value]) 
        a_file.close()
        


# In[2]:


from __future__ import print_function
from IPython import display
get_ipython().system('git clone https://github.com/tensorflow/tpu')
display.clear_output()

# setup path
import sys
sys.path.append("/home/ubuntu/.jupyter/MyNotebooks/tpu/models/official/efficientnet")
sys.path.append("/home/ubuntu/.jupyter/MyNotebooks/tpu/models/common")

model_name = "efficientnet-b4" #@param
print("Done")


# In[3]:


path = name
image_file = name
display.display(display.Image(path))


# In[5]:


import os

# train_dir = './res/'

def progress(percent, width=50):
    '''Progress printing function'''
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # Netsed use of string splicing
    print('\r%s %d%% ' % (show_str, percent), end='')

def is_valid_jpg(jpg_file):
    with open(jpg_file, 'rb') as f:
        f.seek(-2, 2)
        buf = f.read()
        f.close()
        return buf ==  b'\xff\xd9'  # Determine whether the .jpg contains the end field

myFile = open("corrupted_images.txt", 'a')
fDir = '/home/ubuntu/.jupyter/MyNotebooks/imagenet-a/imagenet-a/'
# fSubDir = ["n01498041/", "n01531178/", "n01534433/"]
fSubDir = os.listdir(fDir)

for i in range(0, len(fSubDir)):
  train_dir = fDir + fSubDir[i] + '/'

  data_size = len([lists for lists in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, lists))])
  recv_size = 0
  incompleteFile = 0
  print('file tall : %d' % data_size)

  for file in os.listdir(train_dir):
      if os.path.splitext(file)[1].lower() == '.jpg':
          ret = is_valid_jpg(train_dir + file)
          if ret == False:
              incompleteFile = incompleteFile + 1
              print(train_dir + file + '\n')
              myFile.write(train_dir + file + '\n')
              os.remove(train_dir + file)

      recv_per = int(100 * recv_size / data_size)
      progress(recv_per, width=30)
      recv_size = recv_size + 1

  progress(100, width=30)
  print('\nincomplete file : %d' % incompleteFile)
myFile.close()


# In[3]:


import  eval_ckpt_main as eval_ckpt
import tensorflow.compat.v1 as tf
import sys, json
import time


from random import seed
from random import randint
# seed random number generator
seed(1)

# ----------------------.......----------------------
def randomSample(imagedir):
  if (len(imagedir) <= 6):
      return imagedir   
  else:
      newimagedir = [] 
      for i in range (0, 5):
          value = randint(0, len(imagedir) - 1)
          newimagedir.append(imagedir[value]) 
      return newimagedir  
    
# ----------------------.......----------------------
def isClassFound(folderLabel, possibleClasses):
    isFound = False
    if folderLabel in possibleClasses: 
          isFound = True
    return isFound

# ----------------------.......----------------------
def calculateTopAccuracy(topAcc, totalNumPic):
  topAccValues = [0.0, 0.0, 0.0, 0.0, 0.0]
  for i in range(0, len(topAcc)): 
    topAccValues[i] = topAcc[i]/totalNumPic 
  return topAccValues

# ----------------------.......----------------------
def printResult(model_name, topAcc, topAccValues, totalNumPic):
    print("Model name: " + model_name)
    print("Top Accuracy" + "\t" + "Correctly Predicted Ones" + "\t" + "TotalPic" + "\t" + "TotalAccVal") 
    for i in range(0, len(topAcc)): 
      print(i+1, "\t\t", topAcc[i], "\t\t\t\t", totalNumPic, "\t\t", topAccValues[i])



# ----------------------.......----------------------
get_ipython().system('wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/noisystudent/{model_name}.tar.gz -O {model_name}.tar.gz')
get_ipython().system('tar xf {model_name}.tar.gz')
ckpt_dir = model_name
# !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt -O labels_map.txt
get_ipython().system('wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.json -O labels_map.json')
labels_map_file = "/home/ubuntu/.jupyter/MyNotebooks/labels_map.json"
eval_driver     = eval_ckpt.get_eval_driver(model_name)

imagepath = '/home/ubuntu/.jupyter/MyNotebooks/imagenet-a/imagenet-a/'
imagedirs=os.listdir(imagepath)
# ----------------------.......DESCRIPTION.......----------------------
totalNumofPic = 0   
topAcc        = [0, 0, 0, 0, 0]

for i in range(0, len(randdirs3)):
    file1 = open("noisyresults3"+model_name+".txt", "a")
    start_time = time.time()
    mainpath = randdirs3[i]
    print(randdirs3[i][56:65])
    totalNumofPic += 1
    image_files = [mainpath] # Critical 
    print(mainpath)
    pred_idx, pred_prob  = eval_driver.eval_example_images(ckpt_dir, image_files, labels_map_file) # Critical
    #print(pred_idx)
    folderNumber = randdirs3[i][56:65]     
    # Find Folder label in Imagenet-A 
    folderLabel  = folders[folderNumber] 
    print(folderLabel)
    isFound  = False
    classnumArr = pred_idx[0]
    # print("Array is of type: ", type(classnumArr))
    # classnumArr[2] = int("6")
    # print(classnumArr)
    for k in range (0, 5): 
      classnum = classnumArr[k] # It works 
      # print(classnum)
      possibleClasses = d[int(classnum)]
      # print(possibleClasses)
      if isFound == False:
        isFound      = isClassFound(folderLabel, possibleClasses)
        if isFound:
          topAcc[k] += 1    
      else: 
        topAcc[k]   += 1
    print(topAcc)
    print("--- %s seconds ---" % (time.time() - start_time))
    file1.write(mainpath+" "+ str(totalNumofPic) + " ")
    file1.write(str(topAcc))
    file1.write("\n")
    file1.close()


file1 = open("noisyresults3"+model_name+".txt", "a") 
file1.write("Model name: " + model_name + "\n")
file1.write("Top Accuracy" + "\t" + "Correctly Predicted Ones" + "\t" + "TotalPic" + "\t" + "TotalAccVal" + " \n") 
for i in range(0, len(topAcc)): 
    file1.write(str(i+1)+ "\t\t"+ str(topAcc[i])+ "\t\t\t\t"+ str(totalNumofPic)+ "\n")
file1.close()


# In[ ]:


topAcc       = [123, 123, 123, 345, 567]
totalNumPic  = 7500

def calculateTopAccuracy(topAcc, totalNumPic):
  topAccValues = [0.0, 0.0, 0.0, 0.0, 0.0]
  for i in range(0, len(topAcc)): 
    topAccValues[i] = topAcc[i]/totalNumPic 
  return topAccValues

topAccValues = calculateTopAccuracy(topAcc, totalNumPic)
print(topAccValues)


# In[ ]:



  #manage excetions here


# In[ ]:


# WORKING SINGLE EXAMPLE EFFICIENTNET_ADVPROP 
import  eval_ckpt_main as eval_ckpt
import tensorflow.compat.v1 as tf
# import tensorflow as tf
import sys, json

get_ipython().system('wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/advprop/{model_name}.tar.gz -O {model_name}.tar.gz')
get_ipython().system('tar xf {model_name}.tar.gz')
ckpt_dir = model_name
# !wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt -O labels_map.txt
get_ipython().system('wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.json -O labels_map.json')

labels_map_file = "/content/labels_map.json"

image_files = [image_file]
eval_driver = eval_ckpt.get_eval_driver(model_name)
# pred_idx, pred_prob = eval_driver.eval_example_images(ckpt_dir, image_files, labels_map_file)
pred_idx, pred_prob  = eval_driver.eval_example_images(ckpt_dir, image_files, labels_map_file)
print(pred_idx)





folderNumber = "n01498041"
# Find true label 
folderLabel  = folders[folderNumber]  
# print(folderTrueLabel)
# Find true label

def isClassFound(folderLabel, possibleClasses):
    isFound = False
    if folderLabel in possibleClasses: 
          isFound = True
    return isFound
    
topAcc = [0, 0, 0, 0, 0] 

isFound  = False
classnumArr = pred_idx[0]
print("Array is of type: ", type(classnumArr))
classnumArr[2] = int("6")
print(classnumArr)
for i in range (0, 5): 
  # classnum        = pred_idx[i]
  classnum = classnumArr[i] # It works 
  # print(classnum)
  possibleClasses = d[int(classnum)]
  print(possibleClasses)
  if isFound == False:
      isFound     =  isClassFound(folderLabel, possibleClasses)
      if isFound:
        topAcc[i] += 1  
        # 1. İndekste bulundu.  
  else: 
      topAcc[i] += 1 

print(topAcc)
  


