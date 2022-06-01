#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Introduction

Breast cancer is a widely occurring cancer in women worldwide and is related to high mortality. 
We are taking into account in this project is Invasive Ductal Carcinoma (IDC), since it is the most common subtype of all breast cancers.
Accurately identifying and categorizing breast cancer subtypes is an important clinical task, 
and automated methods can be used to save time and reduce error.

What does our job consists of? 

We will use Machine Learning methods to classify benign and malignant Invasive Ductal Carcinoma from histopathology images.

Interest

This topic was especially in our interest because in this case, machine learning algorithm and their performance
can be used directly to influence a person's life. We are especially interested in how can machines improve people's life
by improving diagnosis in terms of accuracy and time. These two elements are esential when it comes to life-threataning diseases like cancer
and a faster diagnosis, like for example capturing cancer on the early stages, has a higher possibilty of saving the patient's life.
There is still a lot of work to be done in making accurate, precise machine learning algorithm that can be used in the field of Medicine 
and we believe that paying close attention to these problems and trying to improve the algorithms day after day, might really be 
life-changing for humans and humankind.


# In[ ]:


Dataset

The dataset comes from a 2016 study - "Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases" by Andrew Janowczyk and Anant Madabhushi. 
Their study focused on several tasks, one of which was IDC clasification, for which they had an F-score of 0.7648 on 50k testing patches.

The dataset we're working with is derived from 279 patients, each of which has a unique ID. Each patient has a dedicated folder, named by their ID, with two subfolders - 0 and 1. 
The folder named 0 consists of images of benign tissue samples (those without IDC markers). 
The folder named 1 consists of images of malignant tissue samples (those containing IDC markers).

Histopathology images are large, and very small features and markers are present, which is why the images were brokend down into patches, 50x50 pixels in size. 
Each patient, therefore, has many image patches, that together would comprise entire images.
Each patch has a distinct name format - uxXyYclassC.png, where u is the patient's ID, x is the X-coordinate from which the patch was extracted, 
y is the Y-coordinate from which the patch was extracted and the class is either 0 or 1, denoting whether IDC markers are present or not in that patch


# In[ ]:


from IPython import display
display.Image("/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/download\ \(3\).png")


# In[75]:


pip install keras-tuner --upgrade


# In[85]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from numpy.random import permutation
from matplotlib.colors import ListedColormap
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import keras.backend as K


# In[ ]:


Firstly, we upload the dataset. 
Secondly, we think a good idea to start this project is to apply exploratory data methods in order to become
more familiar with the chosen dataset.


# In[2]:


data = os.listdir("/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive")


# In[129]:



img1_0 = Image.open('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive/8863/0/8863_idx5_x101_y1251_class0.png')
img2_0 = Image.open('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive/8863/0/8863_idx5_x101_y1301_class0.png')
img3_0 = Image.open('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive/8863/0/8863_idx5_x151_y1151_class0.png')
img1_1 = Image.open('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive/8863/1/8863_idx5_x1551_y951_class1.png')
img2_1 = Image.open('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive/8863/1/8863_idx5_x1001_y801_class1.png')
img3_1 = Image.open('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive/8863/1/8863_idx5_x1001_y1501_class1.png')


f, ax = plt.subplots(2,3)
ax[0,0].imshow(img1_0)
ax[0,1].imshow(img2_0)
ax[1,0].imshow(img1_1)
ax[1,1].imshow(img2_1)
ax[0,2].imshow(img3_0)
ax[1,2].imshow(img3_1)

print(f)


# In[130]:


patient_8863 = os.listdir("/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive/8863")
patient_8863_0 = os.listdir("/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive/8863/0")
patient_8863_1 = os.listdir("/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive/8863/1")


# In[131]:


print(patient_8863_0[0])


# In[132]:


df_0 = pd.DataFrame()
listOfFiles = [patient_8863_0, patient_8863_1]
idxNumber = 0
for i in listOfFiles:
    for path in i:
        split = path.split('_')
        patient_id = split[0]
        x_coord = split[2][1:]
        y_coord = split[3][1:]
        idc_class = split[4][5]
        
        data = {"index": [idxNumber],
                "patient_id": [patient_id],
                "idc_class": [idc_class],
                "x_coord": [x_coord],
                "y_coord": [y_coord],
                "path": ["/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive/8863/"+ idc_class + "/"+path]}
        data_df = pd.DataFrame(data)
        idxNumber += 1
        df_0 = pd.concat([df_0, data_df])
        
df_8863 = df_0.set_index("index")


# In[133]:


df_8863


# In[134]:


df_8863['x_coord'] = df_8863['x_coord'].astype('int')
df_8863['y_coord'] = df_8863['y_coord'].astype('int')
df_8863['idc_class'] = df_8863['idc_class'].astype('int')


# In[227]:


colors = ['pink', 'purple']
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(x = df_8863['x_coord'], y=df_8863['y_coord'], c=df_8863['idc_class'],cmap=ListedColormap(colors), s=30, marker='s')
ax.set_title("Location of the patches in the whole image from patient 8863")
ax.set_xlabel("y coord")
ax.set_ylabel("x coord")
ax.invert_yaxis()
#ax.set_ylim(bottom=0)
plt.show()


# In[139]:


max_point = [df_8863['x_coord'].max(), df_8863['y_coord'].max()]

grid = 255*np.ones(shape = (max_point[1] + 50, max_point[0] + 50, 3)).astype(np.uint8)
mask = 255*np.ones(shape = (max_point[1] + 50, max_point[0] + 50, 3)).astype(np.uint8)


# In[140]:


for i in range(len(df_8863)):
        # Get image and label
    OldImage = cv2.imread(df_8863['path'][i])
    idc_class = df_8863['idc_class'][i]
        
    dim = (50, 50)
        
    image = cv2.resize(OldImage, dim, interpolation = cv2.INTER_AREA)
        # Extract X and Y coordinates
    x_coord = df_8863['x_coord'][i]
    y_coord = df_8863['y_coord'][i]
        # Add 50 pixels to find ending boundary for each image
    x_end = x_coord + 50
    y_end = y_coord + 50
        
        
        # `grid` will then contain each patch's image values encoded into the grid
    grid[y_coord:y_end, x_coord:x_end] = image
        
        # If `idc_class` is `1`, change the RED channel of the `mask` to 255 (intense red)
        # and other channels to `0` (remove color info, leaving just red)
    if idc_class == 1:
        mask[y_coord:y_end, x_coord:x_end, 0] = 255
        mask[y_coord:y_end, x_coord:x_end, 1:] = 0


# In[224]:


fig, ax = plt.subplots(1,2,figsize=(20,10))
plt.gca().invert_yaxis()

ax[0].imshow(grid, alpha=0.8)
ax[0].set_xlabel("y-coord")
ax[0].set_ylabel("y-coord")
ax[1].imshow(mask, alpha=0.8)
ax[1].imshow(grid, alpha=0.8)
ax[1].grid(False)
ax[1].set_xlabel("y-coord")
ax[1].set_ylabel("y-coord")
#ax.set_ylim(bottom=0)
#ax[1].set_ylim(bottom=0)

ax[0].set_title("Breast tissue of patient 8863")
ax[1].set_title("Cancer tissue in the breast tissue of patient 8863")


# In[13]:


data = glob('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/archive/**/*.png', recursive=True)


# In[ ]:


We've worked with DataFrames so far, though, this was all without images - we only stored their paths in case we want to retrieve and plot them.
One way to load images is to simply iterate through the data and load them in:


# In[199]:


df_0 = pd.DataFrame()
idxNumber = 0
for path in data:
    split = path.split('_')
    # Extract elements 2 and 3, substringing the first char
    patient_id = split[7].split('/')[-1]
    x_coord = split[9][1:]
    y_coord = split[10][1:]
    idc_class = split[11][5]
    
    df_data = {"idc_class": [idc_class],
               "patient_id": [patient_id],
               "x_coord": [x_coord],
               "y_coord": [y_coord],
               "path": [path],
               "index": [idxNumber]}
    idxNumber += 1
    data_df = pd.DataFrame(df_data)
    df_0 = pd.concat([df_0, data_df])
    df = df_0.set_index("index")


# In[46]:


df = df_0.set_index("index")


# In[186]:


df['patient_id'] = df['patient_id'].astype('int')
df['x_coord'] = df['x_coord'].astype('int')
df['y_coord'] = df['y_coord'].astype('int')
df['idc_class'] = df['idc_class'].astype('int')


# In[ ]:


Here I create directories to save my dataframe, in order not to run it every time because it takes a lot of time


# In[48]:


os.makedirs('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/DataFrame', exist_ok=True)
df.to_csv('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/DataFrame/all.csv') 


# In[ ]:


We're creating a truncated dataset to test out the models on smaller sets for efficiency's sake. 
You're free to use the entirety of the dataset, 
but be prepared to wait a long time before you can benchmark them.
Once the benchmarking is done on smaller datasets, 
we can load in the entirety of the images.


# In[14]:


path = '/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/DataFrame/all.csv'
df = pd.read_csv(path)


# In[15]:


df = df.set_index("index")


# In[16]:


df


# In[187]:


fig, ax = plt.subplots(figsize=(10,5))
sns.histplot(df.groupby("patient_id").size(), color="Orange", kde=False, bins=30)
ax.set_xlabel("Number of patches")
ax.set_ylabel("Frequency")
ax.set_title("How many patches do we have per patient?")


# In[188]:


fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(data = df, x ="idc_class", palette='PRGn', ax=ax);
ax.set_ylabel("Count")
ax.set_xlabel("Classes")
ax.set_title("Count of each category")


# In[33]:


shuffled = df.iloc[permutation(df.index)]
print(shuffled.head())


# In[34]:


x = []
y = []

# Loading in 1000 images
for i in shuffled["path"][:1000]:
    if i.endswith('.png'):
        label=i[-5]
        img = cv2.imread(i)
        # Transformation steps, such as resizing
        img = cv2.resize(img,(200,200))
        x.append(img)
        y.append(label)


# In[35]:


x = np.array(x, dtype='float16')
y = np.array(y, dtype='float16')


X_train, X_test, y_train, y_test = train_test_split(x,y, shuffle=True, test_size=0.3)


# In[38]:


if not os.path.exists('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/'):
    os.mkdir('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/')

    os.mkdir('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/train/')
    os.mkdir('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/test/')

    os.mkdir('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/train/0/')
    os.mkdir('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/train/1/')
    os.mkdir('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/test/0/')
    os.mkdir('/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/test/1/')
    
    
    


# In[55]:


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


# In[ ]:


Now, let's iterate over the length of the dataset, in large steps, and use the steps as the starting and ending indices for our data list, 
loading the associated images in, reshaping them, and saving them in the appropriate folder:


# In[ ]:


for batch_num, indices in enumerate(range(1000, int(len(df)/10), 1000), 1):
    x = []
    y = []
    
    for i in df["path"][indices-1000:indices]:
        if i.endswith('.png'):
            label=i[-5]
            img = cv2.imread(i)
            img = cv2.resize(img,(100,100))
            x.append(img)
            y.append(label)
        
    x = np.array(x, dtype='float32')
    y = np.array(y, dtype='float32')
    
    X_train, X_test, y_train, y_test = train_test_split(x,y, shuffle=True, test_size=0.2)
    
    for index, img in enumerate(X_train):
        random_value = tf.random.uniform(())
        if random_value > 0.5 and random_value < 0.65:
            img = tf.image.flip_left_right(img).numpy()
        if random_value > 0.6 and random_value < 0.75:
            img = tf.image.flip_up_down(img).numpy()
        cv2.imwrite(f"/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/train/{int(y_train[index])}/batch{batch_num}_sample{index}.png", img.astype('int'))
    
    for index, img in enumerate(X_test):
        cv2.imwrite(f"/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/test/{int(y_test[index])}/batch{batch_num}_sample{index}.png", img.astype('int'))
        
        
        
        


# In[256]:



path_list = glob("/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/train/**/*.png")

trainingData_x = []
trainingData_y = []

for path in path_list:
    if path.endswith('.png'):
        label = int(path.split('/')[-2])
        img = cv2.imread(path)
        img = cv2.resize(img,(200,200))
        trainingData_x.append(img)
        trainingData_y.append(label)

trainingData_x = np.array(trainingData_x, dtype='float32')
trainingData_y = np.array(trainingData_y, dtype='float32')

X_train, X_valid, y_train, y_valid = train_test_split(trainingData_x, trainingData_y, shuffle=True, test_size=0.1, random_state=42)

X_train = np.array(X_train, dtype='float32')
X_valid = np.array(X_valid, dtype='float32')

path_list = glob("/Users/carolina/Documents/Semester_8/Introduction_to_Machine_Learning_and_Data_Mining/Project/hist_images_truncated/test/**/*.png")

testData_x = []
testData_y = []

for path in path_list:
    if path.endswith('.png'):
        label = int(path.split('/')[-2])
        img = cv2.imread(path)
        img = cv2.resize(img,(200,200))
        testData_x.append(img)
        testData_y.append(label)

X_test = np.array(testData_x, dtype='float32')
y_test = np.array(testData_y, dtype='float32')





# In[216]:


fig, ax = plt.subplots(1,3,figsize=(20,5))
sns.countplot(x = y_train, ax=ax[0], palette="Purples")
ax[0].set_title("Train data")
sns.countplot(x = y_valid, ax=ax[1], palette="Oranges")
ax[1].set_title("Dev data")
sns.countplot(x= y_test, ax=ax[2], palette="Greens")
ax[2].set_title("Test data")


# In[ ]:


get_ipython().set_next_input('First, we raise an issue here. Is our class imbalanced');get_ipython().run_line_magic('pinfo', 'imbalanced')
Given the fact that negative samples are much more numerous than positive ones 
- our test set will also have a lot of negative samples. 
Since there's 277524 samples in total, 198738 of which are negative - that's ~71% class 0 samples.

In conclusion: accuracy is not a good metric to use when you have class imbalance.


# In[ ]:


Accuracy is a metric for classification models that measures the number of predictions that are correct
as a percentage of the total number of predictions that are made. 
As an example, if 90% of your predictions are correct, your accuracy is simply 90%.

Accuracy is a useful metric only when you have an equal distribution of classes on your classification.
This means that if you have a use case in which you observe more data points of one class than of another,
the accuracy is not a useful metric anymore.

get_ipython().set_next_input('But how to fix this issue');get_ipython().run_line_magic('pinfo', 'issue')

There are different ways like for example:
    1. undersampling
    2. oversampling
    3. SMOTE data augumentation
    4. better accuracy metrics


# In[ ]:


The F1 score: combining Precision and Recall
    
Precision and Recall are the two building blocks of the F1 score.
The goal of the F1 score is to combine the precision and recall metrics into a single metric. 
At the same time, the F1 score has been designed to work well on imbalanced data.
Another definition for F1 score is the harmonic mean of precision and recall.
 
F1=(2∗Precision∗Recall)/(Precision+Recall)


# In[ ]:


Now we will try 3 different models of CNN, chosen from our research online, to be the best and most used
with Image Classification problem, but we will use also f1 as a metric too while running the CNN since our
classes are imbalanced.


# In[ ]:


First that we will try is EfficientNet which is a convolutional neural network architecture and scaling method that 
uniformly scales all dimensions of depth/width/resolution using a compound coefficient.


# In[ ]:


With the custom CNN baseline checked, let's see if we can utilize some of the pre-existing
architectures to boost the performance. 
These architectures have specialized building blocks and will typically perform better 
than a solution as simple as the one laid out in the previous section.
Keras comes with a bunch of built-in models, both pre-trained and empty!


# In[ ]:


The EfficientNet family, spanning from B0 to B7 is an efficiently-scaling, highly-performant family of models.
Alongside some other architectures, it's consistently ranked in the top performant models in most benchmarks. 
There's a high probability that it will perform on this task well, so let's start out with EfficientNetB0!


# In[ ]:


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    
    return K.mean(f1)



# In[ ]:


def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    
    return 1 - K.mean(f1)



# In[59]:


model_efficientNetB0 = keras.models.Sequential([
    keras.applications.EfficientNetB0(input_shape=(200,200,3),weights='imagenet',include_top=False), 
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_efficientNetB0.summary()


# In[ ]:





# In[207]:


callbacks = [tf.keras.callbacks.EarlyStopping(patience=5),
             tf.keras.callbacks.ModelCheckpoint(filepath='breast_cancer_effnetb0.h5', save_best_only=True)]

model_efficientNetB0.compile(loss="binary_crossentropy",
                             optimizer='adam',
                             metrics=[f1,
                                      keras.metrics.BinaryAccuracy(),
                                      keras.metrics.Precision(),
                                      keras.metrics.Recall(),
                                      keras.metrics.AUC()])

history_efficientNetB0 = model_efficientNetB0.fit(X_train, y_train,
                                    validation_data = (X_valid, y_valid),
                                    callbacks = callbacks,
                                    epochs = 15)

model_efficientNetB0.save('model_efficientNetB0.h5')



# In[ ]:


model_efficientNetB0 = tf.keras.models.load_model("model_efficientNetB0.h5")


# In[61]:


losse_efficientNetB0 = pd.DataFrame(model_efficientNetB0.history.history)
losse_efficientNetB0


# In[39]:


fig, ax = plt.subplots(3, 2, figsize=(10,10))
ax[0,0].plot(losse_efficientNetB0["loss"])
ax[0,0].plot(losse_efficientNetB0["val_loss"])
ax[0,0].legend(['train_data','validation_data'], loc='upper left')
ax[0,0].set_title('loss analysis')

ax[0,1].plot(losse_efficientNetB0["binary_accuracy"])
ax[0,1].plot(losse_efficientNetB0["val_binary_accuracy"])
ax[0,1].legend(['train_data','validation_data'], loc='upper left')
ax[0,1].set_title('binary_accuracy analysis')

ax[1,0].plot(losse_efficientNetB0["precision_4"])
ax[1,0].plot(losse_efficientNetB0["val_precision_4"])
ax[1,0].legend(['train_data','validation_data'], loc='upper left')
ax[1,0].set_title('precision analysis')

ax[1,1].plot(losse_efficientNetB0["recall_4"])
ax[1,1].plot(losse_efficientNetB0["val_recall_4"])
ax[1,1].legend(['train_data','validation_data'], loc='upper left')
ax[1,1].set_title('recall analysis')

ax[2,0].plot(losse_efficientNetB0["auc_4"])
ax[2,0].plot(losse_efficientNetB0["val_auc_4"])
ax[2,0].legend(['train_data','validation_data'], loc='upper left')
ax[2,0].set_title('auc analysis')

# ax[2,1].plot(losse_efficientNetB0["f1"])
# ax[2,1].plot(losse_efficientNetB0["val_f1"])
# ax[2,1].legend(['train_data','validation_data'], loc='upper left')
# ax[2,1].set_title('fit analysis')


# In[ ]:


Now, after running the cnn without using f1 as a metric or as a loss function, let's try it with these matrics.


# In[62]:


model_effNetB0_f1 = keras.models.Sequential([
    keras.applications.EfficientNetB0(input_shape=(200,200,3),weights='imagenet',include_top=False), 
    keras.layers.BatchNormalization(),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_effNetB0_f1.summary()



# In[64]:


callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath='breast_cancer_effnetb0_f1_loss.h5', save_best_only=True, monitor="f1_val")]

model_effNetB0_f1.compile(loss=f1_loss,
                          optimizer='adam',
                          metrics=[f1,
                                   keras.metrics.BinaryAccuracy(),
                                   keras.metrics.Precision(),
                                   keras.metrics.Recall(),
                                   keras.metrics.AUC()])

history_effNetB0_f1 = model_effNetB0_f1.fit(X_train, y_train,
                                            validation_data=(X_valid, y_valid),
                                            callbacks=callbacks,
                                            epochs=15)

model_effNetB0_f1.save('model_efficientNetB0_f1.h5')


# In[ ]:


model_effNetB0_f1 = tf.keras.models.load_model("model_efficientNetB0_f1.h5")


# In[ ]:


losse_effNetB0_f1 = pd.DataFrame(model_effNetB0_f1.history.history)
losse_effNetB0_f1


# In[220]:


sns.set_theme()
fig, ax = plt.subplots(2, 2, figsize=(10,10))
ax[0,0].plot(losse_effNetB0_f1["loss"])
ax[0,0].plot(losse_effNetB0_f1["val_loss"])
ax[0,0].legend(['train_data','validation_data'], loc='lower right')
ax[0,0].set_title('F1-score Loss')
ax[0,0].set_ylabel("value")

#ax[0,1].plot(losse_effNetB0_f1["binary_accuracy"])
#ax[0,1].plot(losse_effNetB0_f1["val_binary_accuracy"])
#ax[0,1].legend(['train_data','validation_data'], loc='upper left')
#ax[0,1].set_title('binary_accuracy analysis')

ax[1,0].plot(losse_effNetB0_f1["precision_3"])
ax[1,0].plot(losse_effNetB0_f1["val_precision_3"])
ax[1,0].legend(['train_data','validation_data'], loc='lower left')
ax[1,0].set_title('Precision')
ax[1,0].set_xlabel("Number of epochs")
ax[1,0].set_ylabel("value")


ax[1,1].plot(losse_effNetB0_f1["recall_4"])
ax[1,1].plot(losse_effNetB0_f1["val_recall_4"])
ax[1,1].legend(['train_data','validation_data'], loc='lower left')
ax[1,1].set_title('Recall')
ax[1,1].set_xlabel("Number of epochs")

#ax[2,0].plot(losse_effNetB0_f1["auc_4"])
#ax[2,0].plot(losse_effNetB0_f1["val_auc_4"])
#ax[2,0].legend(['train_data','validation_data'], loc='upper left')
#ax[2,0].set_title('auc analysis')

ax[0,1].plot(losse_effNetB0_f1["f1"])
ax[0,1].plot(losse_effNetB0_f1["val_f1"])
ax[0,1].legend(['train_data','validation_data'], loc='upper left')
ax[0,1].set_title('F1-score')


# In[74]:


evaluation_effNetB0_f1 = model_effNetB0_f1.evaluate(X_test, y_test)


# In[86]:


pred_effNetB0 = model_effNetB0_f1.predict(X_test)


# In[88]:


sns.heatmap(confusion_matrix(y_true = y_test, y_pred = K.round(pred_effNetB0)), annot=True, fmt='g')
plt.show()


# In[65]:


model_resnet = keras.models.Sequential([
    keras.applications.ResNet50(input_shape=(200,200,3), weights='imagenet', include_top=False), 
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_resnet.summary()


# In[66]:


callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath='breast_cancer_resnet50.h5',  save_best_only=True, monitor="f1_val")]

model_resnet.compile(loss=f1_loss,
              optimizer='adam',
              metrics=[f1,
                  keras.metrics.BinaryAccuracy(),
                  keras.metrics.Precision(),
                  keras.metrics.Recall(),
                  keras.metrics.AUC()
              ])

history_resnet = model_resnet.fit(X_train, y_train,
                     validation_data=(X_valid, y_valid),
                     callbacks=callbacks,
                     epochs=15)

model_resnet.save('model_resnet.h5')


# In[ ]:


model_resnet = tf.keras.models.load_model("model_resnet.h5")


# In[76]:


losse_resnet = pd.DataFrame(model_resnet.history.history)
losse_resnet


# In[222]:


fig, ax = plt.subplots(2, 2, figsize=(10,10))
ax[0,0].plot(losse_resnet["loss"])
ax[0,0].plot(losse_resnet["val_loss"])
ax[0,0].legend(['train_data','validation_data'], loc='upper left')
ax[0,0].set_title('F1-score Loss')
ax[0,0].set_ylabel("value")

#ax[0,1].plot(losse_resnet["binary_accuracy"])
#ax[0,1].plot(losse_resnet["val_binary_accuracy"])
#ax[0,1].legend(['train_data','validation_data'], loc='lower left')
#ax[0,1].set_title('binary_accuracy analysis')

ax[1,0].plot(losse_resnet["precision_4"])
ax[1,0].plot(losse_resnet["val_precision_4"])
ax[1,0].legend(['train_data','validation_data'], loc='lower left')
ax[1,0].set_title('Precision')
ax[1,0].set_xlabel("Number of epochs")
ax[1,0].set_ylabel("value")

ax[1,1].plot(losse_resnet["recall_5"])
ax[1,1].plot(losse_resnet["val_recall_5"])
ax[1,1].legend(['train_data','validation_data'], loc='lower left')
ax[1,1].set_title('Recall')
ax[1,1].set_xlabel("Number of epochs")

#ax[2,0].plot(losse_resnet["auc_5"])
#ax[2,0].plot(losse_resnet["val_auc_5"])
#ax[2,0].legend(['train_data','validation_data'], loc='lower left')
#ax[2,0].set_title('auc analysis')

ax[0,1].plot(losse_resnet["f1"])
ax[0,1].plot(losse_resnet["val_f1"])
ax[0,1].legend(['train_data','validation_data'], loc='lower left')
ax[0,1].set_title('F1-score')


# In[ ]:





# In[84]:


evaluation_resnet = model_resnet.evaluate(X_test, y_test)


# In[87]:


pred_resnet = model_resnet.predict(X_test)


# In[89]:


sns.heatmap(confusion_matrix(y_true = y_test, y_pred = K.round(pred_resnet)), annot=True, fmt='g')
plt.show()


# In[ ]:


Our next model would be a model whose author is the author of Keras itself. It is a high-performance architecture 
and it is called: "Google's Xception". We chose an architecture with which we haven't worked so far. 
Let's see how it will perform:


# In[67]:


model_xception = keras.models.Sequential([
    keras.applications.Xception(input_shape=(200,200,3),weights='imagenet',include_top=False), 
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model_xception.summary()


# In[68]:


callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath='breast_cancer_xception.h5',  save_best_only=True, monitor="f1_val")]

model_xception.compile(loss=f1_loss,
                       optimizer='adam',
                       metrics=[f1,
                                keras.metrics.BinaryAccuracy(),
                                keras.metrics.Precision(),
                                keras.metrics.Recall(),
                                keras.metrics.AUC()])

history_xception = model_xception.fit(X_train, y_train,
                     validation_data=(X_valid, y_valid),
                     callbacks=callbacks,
                     epochs=15)

model_xception.save('model_xception.h5')


# In[ ]:


model_xception = tf.keras.models.load_model("model_xception.h5")


# In[90]:


losse_xception = pd.DataFrame(model_xception.history.history)
losse_xception


# In[221]:


fig, ax = plt.subplots(2, 2, figsize=(10,10))
ax[0,0].plot(losse_xception["loss"])
ax[0,0].plot(losse_xception["val_loss"])
ax[0,0].legend(['train_data','validation_data'], loc='upper left')
ax[0,0].set_title('F1-score Loss')
ax[0,0].set_ylabel("value")

#ax[0,1].plot(losse_xception["binary_accuracy"])
#ax[0,1].plot(losse_xception["val_binary_accuracy"])
#ax[0,1].legend(['train_data','validation_data'], loc='lower left')
#ax[0,1].set_title('binary_accuracy analysis')

ax[1,0].plot(losse_xception["precision_5"])
ax[1,0].plot(losse_xception["val_precision_5"])
ax[1,0].legend(['train_data','validation_data'], loc='lower left')
ax[1,0].set_title('Precision')
ax[1,0].set_xlabel("Number of epochs")
ax[1,0].set_ylabel("value")

ax[1,1].plot(losse_xception["recall_6"])
ax[1,1].plot(losse_xception["val_recall_6"])
ax[1,1].legend(['train_data','validation_data'], loc='lower left')
ax[1,1].set_title('Recall')
ax[1,1].set_xlabel("Number of epochs")

#ax[2,0].plot(losse_xception["auc_6"])
#ax[2,0].plot(losse_xception["val_auc_6"])
#ax[2,0].legend(['train_data','validation_data'], loc='lower left')
#ax[2,0].set_title('auc analysis')

ax[0,1].plot(losse_xception["f1"])
ax[0,1].plot(losse_xception["val_f1"])
ax[0,1].legend(['train_data','validation_data'], loc='lower left')
ax[0,1].set_title('F1-score')


# In[93]:


evaluation_xception = model_xception.evaluate(X_test, y_test)


# In[94]:


pred_xception = model_xception.predict(X_test)


# In[95]:


sns.heatmap(confusion_matrix(y_true = y_test, y_pred = K.round(pred_xception)), annot=True, fmt='g')
plt.show()


# In[276]:


fig = plt.figure(figsize=(10,10))

images = X_test
labels = y_test

for index, image in enumerate(images):
    ax = fig.add_subplot(5,5,index+1)
    plt.imshow(image.astype('int'))
    
    image = np.expand_dims(image, 0)
    pred = model_xception.predict(image)
    pred = np.squeeze(pred)
    label = labels[index]
    
    ax.set_title(f'Proba: {np.format_float_scientific(pred, precision=3)}% \n Rounded Pred: {np.round(pred)} \n Actual: {label}')
    
    
plt.tight_layout()
plt.show()


# In[ ]:


**Choosing The Best Model**


# In[ ]:


get_ipython().set_next_input('How to choose');get_ipython().run_line_magic('pinfo', 'choose')

We are interested in the model with the biggest recall but biggest precision also. Usually when one is big, 
the other one is small giving the recall-precision trade-off. Let's start evaluating.


# In[96]:


effnet_recall, effnet_precision, effnet_f1, effnet_params = evaluation_effNetB0_f1[4], evaluation_effNetB0_f1[3], evaluation_effNetB0_f1[1], model_effNetB0_f1.count_params()
resnet_recall, resnet_precision, resnet_f1, resnet_params = evaluation_resnet[4], evaluation_resnet[3], evaluation_resnet[1], model_resnet.count_params()
xception_recall, xception_precision, xception_f1, xcecption_params = evaluation_xception[4], evaluation_xception[3], evaluation_xception[1], model_xception.count_params()


# In[97]:


values = {
    'EffNetB0' : [effnet_recall, effnet_precision, effnet_f1],
    'ResNet50' : [resnet_recall, resnet_precision, resnet_f1],
    'Xception' : [xception_recall, xception_precision, xception_f1]}


# In[101]:


df_performance = pd.DataFrame(values, index = ["Recall", "Precision", "F1-score"]).T


# In[102]:


df_performance


# In[ ]:


As we can see, in our case, EFfNetB0 and Xception are the ones that perform the best. Here we should make a choice
between higher precision or higher recall because EffNet has a higher recall while Xception has a higher Precision.
On the other side, from our previous knowledge we expected ResNet50 to perform much better than it actually does.
In our case Resnet perform very bad with very low Recall and Precision.


# In[ ]:


Also, we will create a bar plot below in order for the values to be more visible and to be convinced for our choice.


# In[208]:


import matplotlib.colors as mcolors
sns.set_theme(style="whitegrid", palette="Paired")


# In[209]:


fig, ax = plt.subplots(figsize=(12, 8))

df_bar = df_performance.reset_index().melt(id_vars=["index"])
ax = sns.barplot(x="variable", y="value", hue="index", data=df_bar)
ax.set(xlabel='metric', ylabel='value')
plt.legend(title = "Model")
print(ax)


# In[268]:


import keras_tuner as kt


# In[269]:


def model_builder(hp):
    model = keras.Sequential([keras.applications.Xception(input_shape=(200,200,3),weights='imagenet',include_top=False), 
                              keras.layers.GlobalAveragePooling2D(),
                              keras.layers.Dense(hp.Int('units', min_value=16, max_value=256, step=32), 
                                                 activation=hp.Choice('activation', ['relu', 'swish'])),
                              keras.layers.Dense(1, activation='sigmoid')])
    
    model.compile(loss=f1_loss,
                  optimizer=hp.Choice('optimizer', ['adam', 'sgd', 'nadam']))

    return model


# In[281]:


tuner = kt.RandomSearch(hypermodel = model_builder,
                        objective='val_loss',
                        max_trials=3b)
    
tuner.search(X_train, y_train, 
             validation_data=(X_valid, y_valid), 
             epochs=3)



# In[282]:


tuner.results_summary(num_)


# In[291]:


best_model = tuner.get_best_models(num_models=1)


# In[296]:


tuner.get_best_hyperparameters()[0]


# In[ ]:


References:


# In[ ]:


https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

https://www.kaggle.com/code/allunia/breast-cancer

https://www.kaggle.com/code/amerii/breast-cancer-classification-end-to-end

https://www.kaggle.com/code/rejpalcz/best-loss-function-for-f1-score-metric/notebook


# In[ ]:


https://www.analyticsvidhya.com/blog/2020/08/top-4-pre-trained-models-for-image-classification-with-python-code/

https://theaisummer.com/cnn-architectures/

https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/

https://builtin.com/data-science/precision-and-recall

https://medium.com/swlh/hyperparameter-tuning-in-keras-tensorflow-2-with-keras-tuner-randomsearch-hyperband-3e212647778f

https://www.tensorflow.org/tutorials/keras/keras_tuner
    
https://keras.io/guides/keras_tuner/getting_started/

