import os
from random import seed
from random import randint
from os import listdir


# Selecting Random sampled images from ImageNet-A
path3 = '/home/ubuntu/.jupyter/MyNotebooks/imagenet-a/imagenet-a/'      #Change to your path  
dirs=os.listdir(path3)

# From random seed
seed(1)

selected_img = [0] * 7500

directories=[]
count=0
# Traversing directories 
for i in range(0, len(dirs)):
  dirs2=os.listdir(path3+dirs[i])
  for y in range(0, len(dirs2)):
    directories.append(path3+dirs[i]+'/'+dirs2[y])
    count+=1

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