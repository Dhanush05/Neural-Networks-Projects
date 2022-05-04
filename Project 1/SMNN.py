#!/usr/bin/env python
# coding: utf-8

# In[28]:


import matplotlib.pyplot as py
from PIL import Image
import numpy as np
import cv2


# In[29]:


from math import *
import math
from keras.models import Sequential
from keras.layers import Dense
import numpy
import random


# <h1>Step1:</h1>
# <h3>Building the image dataset</h3>

# In[30]:


for i in range(10):
    image = Image.open(str(i)+'.png')
    print(image.format)
    print(image.size)
    print(image.mode)
    #Converting given image to grayscale
    img = image.convert('L')

    bw = img.point(lambda x: 1 if x<195 else 0,'1') #change pixel value to BW scale based the threshold.
    #Save the images to the given directory
    bw.save(str(i)+'.png') 


# In[57]:


#read data from the saved file to check
cv2.imread('out5.png',cv2.IMREAD_UNCHANGED) 


# In[58]:


import cv2
d =list()
# to append all the values.
for i in range(10):
    d.append((cv2.imread('out'+str(i)+'.png', cv2.IMREAD_UNCHANGED)).ravel())


# In[59]:


d


# In[60]:


X=np.row_stack(d.pop() for i in range(10)) #pop values into the dataset - X
y=X     #here we will be training over the same images so y=X


# In[61]:


for i in range(10):
    for j in range(256):
        if X[i][j]==255:
            X[i][j]=1


# In[62]:


X[5]


# <h3>Dataset created</h3>
# <h1>Step-2</h1>
# <h3>Developing a Single Layer Perceptron</h1>

# In[63]:


from keras import regularizers

model = Sequential()
model.add(Dense(128, input_dim = 256, activation='relu'))
model.add(Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam')


# <h1>Step-3</h1>
# <h3>Training the SLP on the dataset of 10 images</h3>

# In[65]:


model.fit(X, y, epochs=800, batch_size=3,verbose = True) 


# In[39]:


#preliminary tests
check=Y[7]
for an in range(256):
    if(check[an]==0):
        check[an]=255
    else:
        check[an]=0


temp3 = np.reshape(X[0],(16,16)) 
imm = Image.fromarray(temp3)
if imm != 'RGB':
    imm = imm.convert('RGB')

imm.show()


# check=Y[5]
# for an in range(256):
#     if(check[an]==0):
#         check[an]=255
#     else:
#         check[an]=0
# 
# 
# temp3 = np.reshape(X[0],(16,16)) 
# imm = Image.fromarray(temp3)
# if imm != 'RGB':
#     imm = imm.convert('RGB')
# 
# imm.show()

# <h1>Step-4</h1>
# <h3>Test SLP on trained data</h3>

# In[66]:


temp_out = model.predict(X) #prediction of model on trained data X

#step-4a
#--Thresholding predicted output--# 
def thresholdcheck(temp_out):
    
    Y = []
    for i in temp_out:
        temp = []
        for j in i:
            if j > 0.5:
                temp.append(1)
            else:
                temp.append(0)

        Y.append(temp)
    return Y
Y=thresholdcheck(temp_out)


# In[67]:


#Step-4b
#Computing metrics Fh, Ffa



def findFh(expected, actual):
    fhvalues = []
    for i in range(len(expected)):
        fh=0
        ffa=0
        for j in range(256):
            if(expected[i][j]==0 and actual[i][j]==0):
                fh+=1

        ffad = cv2.countNonZero(expected[i]) #ones - white
        ffhd = 256-ffad #zeros - black
        
        fhvalues.append(fh/ffhd)
    return fhvalues


def findFfa(expected, actual):
    ffavalues = []
    for i in range(len(expected)):
        ffa=0
        for j in range(256):
            if (actual[i][j] == 0) and (expected[i][j] != 0):
                ffa += 1
        ffhd = cv2.countNonZero(expected[i])
        ffavalues.append(ffa/ffhd)
    return ffavalues

def drawplot(xvalues,yvalues,title):
    plt.scatter(xvalues, yvalues)
    plt.title(title)
    plt.xlabel("Ffa")
    plt.ylabel("Fh")
    plt.show()
    


Fhvalues = findFh(X, Y)
Ffavalues = findFfa(X, Y)



#Step-4c

drawplot(Ffavalues,Fhvalues,"Ffa Vs Fh")




# <h1>Step-5</h1>
# <h3>Add noise to the images</h3>

# In[68]:


#creating noisy images
def Perturb(actualimages, mean, std_dev):
   NoisyImages = []
   
   for i in actualimages:
       Image = i
       
       mean, std_dev = mean, std_dev

       sampleimg = numpy.random.normal(mean, std_dev, 25)

       indexes = random.sample(range(0, 255), 25)
       
       sampleIndex = 0
       
       for j in indexes:
           Image[j] += sampleimg[sampleIndex]
           sampleIndex += 1
       NoisyImages.append(Image)
       
   return thresholdcheck(NoisyImages)


stdDevs = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]

FhArrays = []
FfaArrays = []

for i in stdDevs:
   Noisyimgs = numpy.array(Perturb(X, 0, i))
   
   
   Fhvalues = findFh(Noisyimgs, Y)
   Ffavalues = findFfa(Noisyimgs, Y)

   drawplot(Ffavalues,Fhvalues,"Ffa Vs Fh at Std Dev(noisy) = " + str(i))

   
   
   FhArrays.append(Fhvalues)
   FfaArrays.append(Ffavalues)




# <h1>Step-6</h1>
# <h3>Displaying the data from step5</h3>

# In[69]:


#saving the data of predicted noisy images
#step 6a

from tabulate import tabulate
table = []
for i in range(9):
    newEntry = []
    newEntry.append(str(i))
    for j in range(len(FhArrays)):
        newEntry.append(str(round(FhArrays[i][j], 2)))
        newEntry.append(str(round(FfaArrays[i][j], 2)))
    table.append(newEntry)
print(tabulate(table, headers = ["Image", "Fh 0.001", "Ffa 0.001",
                                 "Fh 0.002", "Ffa 0.002", "Fh 0.003", "Ffa 0.003", 
                                 "Fh 0.005", "Ffa 0.005", "Fh 0.01", "Ffa 0.01", 
                                "Fh 0.02", "Ffa 0.02", "Fh 0.03", "Ffa 0.03", 
                                 "Fh 0.05", "Ffa 0.05", "Fh 0.1", "Ffa 0.1"]))


# In[71]:


#step 6b

#Ffa and fh graph with each standard deviation represented on a log scale
import matplotlib.pyplot as plt



for i in range(len(FhArrays)):
    for j in range(len(FhArrays[i])):
        for k in stdDevs:
            plt.scatter(k, FhArrays[i][j], color='black')
            plt.scatter(k, FfaArrays[i][j], color = 'yellow')
plt.title("Graph of Ffa,Fh vs Noise Standard Deviation for noise-corrupted Alphanumeric Imagery (16x16 px) for Autoassociative Multi-Layer Perceptron")

plt.xscale('log')
plt.xlabel("Gaussian Noise Level (stdev, at 10 pct xsecn)")
plt.xlim([0, 0.1])

plt.ylim([0, 1])
plt.ylabel("Fh and Ffa")

plt.legend(["Black = Fh", "Yellow = Ffa"])

plt.show()


# In[ ]:




