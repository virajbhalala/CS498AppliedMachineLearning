
# coding: utf-8

# ### CIFAR-10 Image Dataset - Principal Component Analysis
# 
# 1. Data Source- https://www.cs.toronto.edu/~kriz/cifar.html
# 2. About Data- The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
# 3. About PCA: https://en.wikipedia.org/wiki/Principal_component_analysis

# In[4]:


import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import sklearn.manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.manifold import MDS



# In[5]:


#load data's code taken from  https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def createPlot(im):
    r = im[0:1024].reshape(32, 32)
    g = im[1024:2048].reshape(32, 32)
    b = im[2048:].reshape(32, 32)
    img = np.dstack((r, g, b))
    plt.imshow(img)
    plt.show()


# In[6]:


a =unpickle(file="../CIFAR_Image_PCA/cifar-10-batches-py/data_batch_1")
b =unpickle(file="../CIFAR_Image_PCA/cifar-10-batches-py/data_batch_2")
c =unpickle(file="../CIFAR_Image_PCA/cifar-10-batches-py/data_batch_3")
d =unpickle(file="../CIFAR_Image_PCA/cifar-10-batches-py/data_batch_4")
e =unpickle(file="../CIFAR_Image_PCA/cifar-10-batches-py/data_batch_5")
f =unpickle(file="../CIFAR_Image_PCA/cifar-10-batches-py/test_batch")

meta =unpickle(file="../CIFAR_Image_PCA/cifar-10-batches-py/batches.meta")
#Concat dictionary

d = dict(a)
d.update(b)
d.update(c)
d.update(d)
d.update(e)
d.update(f)


# In[7]:


d.get(b'data')


# In[8]:


d.get(b'labels')[1]


# In[9]:


d.get(b'data').shape


# In[10]:


#get unique labels/categories
list(set(d.get(b'labels')))


# In[11]:


# 0 airplane
# 1 automobile
# 2 bird
# 3 cat
# 4 deer
# 5 dog
# 6 frog
# 7 horse
# 8 ship
# 9 truck
categories = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
np.array(meta.get(b'label_names'))


# ## Part 1: For each category,
# ###   1. compute the mean image and the first 20 principal components.
# ### 2. Plot the error resulting from representing the images of each category using the first 20 principal components against the category.
# 
# 

# In[12]:


#create a pandas DF
images = pd.DataFrame( np.array(d.get(b'data')))
labels =pd.DataFrame( np.array(d.get(b'labels')))
images["labels"] =labels


# In[13]:


images.head()


# In[14]:


Image.fromarray(np.array(d.get(b'data'))[1][:1024].reshape(32,32)).resize((100, 100))


# In[15]:


Image.fromarray(np.array(d.get(b'data'))[1][1024:2048].reshape(32,32)).resize((100, 100))


# In[16]:


Image.fromarray(np.array(d.get(b'data'))[1][2048:3072].reshape(32,32)).resize((100, 100))


# In[17]:



createPlot(im = np.array(d.get(b'data').astype('uint8'))[1])


# # Part 1

# In[18]:


#group df by labels
groups = images.groupby('labels')


# In[56]:


meanimages = []
centered =[]
pcaError =[]
for i, cat in enumerate(categories):
    group = groups.get_group(i)
    grp =group.drop("labels", axis=1).reset_index(drop=True)
    #Part1
    ##MEAN IMAGE
    m =grp.mean().astype('uint8')
    meanimages.append(m)
    centered.append(grp - m)
    
    ##PCA 20
    pca = PCA(n_components=20)
    pca.fit(grp)    
    
    ##Error
    components = pca.transform(grp)
    inv = pd.DataFrame(pca.inverse_transform(components))
    pcaError.append(int(np.square(grp-inv).sum(axis=1).mean()))


# In[57]:


for i, cat in enumerate(categories):
    print(cat)
    createPlot(meanimages[i])


# In[58]:


plt.bar(categories, pcaError, align='center', alpha=0.5)
plt.ylabel('Error')
plt.xticks(rotation=90)
plt.title('Error of each Categories')
plt.show()


# In[59]:


pcaError


# # Part 2

# In[22]:


similarities = euclidean_distances(meanimages)
pd.DataFrame(similarities)


# In[23]:


mds = MDS(n_components=2, random_state=1, dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(similarities).embedding_
plt.figure()
plt.scatter(pos[:,0], pos[:,1],)
for i in range (0,10):
    xy=(pos[i][0],pos[i][1])
    plt.annotate(categories[i],xy)
plt.show()


# # Part3

# In[65]:


pcaError = np.zeros((10,10))
for i in range(0,10):
    for j in range(i,10):
        groupA = groups.get_group(i)
        grpA =groupA.drop("labels", axis=1).reset_index(drop=True)     
        
        groupB = groups.get_group(j)
        grpB =groupB.drop("labels", axis=1).reset_index(drop=True)
                                                      
        #get PCA of A
        pcaA = PCA(n_components=20)
        pcaA.fit(grpA) 
        
        #get PCA of B
        pcaB = PCA(n_components=20)
        pcaB.fit(grpB) 
        
       
        
        componentA = pcaB.transform(grpA)
        invA = pd.DataFrame(pcaB.inverse_transform(componentA))
        errorA = int(np.square(grpA-invA).sum(axis=1).mean())
        
        componentB = pcaA.transform(grpB)
        invB = pd.DataFrame(pcaA.inverse_transform(componentB))
        errorB = int(np.square(grpB-invB).sum(axis=1).mean())
        
        
        
        pcaError[i][j] = (errorA+errorB)/2
        #mirror diagonally
        pcaError[j][i] = (errorA+errorB)/2

display(pd.DataFrame(pcaError))

mds = MDS(n_components=2, random_state=1, dissimilarity="precomputed", n_jobs=1)
pos = mds.fit(pcaError).embedding_
plt.figure()
plt.scatter(pos[:,0], pos[:,1])
for i in range (0,10):
    xy=(pos[i][0],pos[i][1])
    plt.annotate(categories[i],xy)
plt.show()

