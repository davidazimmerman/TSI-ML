# %%
#!/usr/bin/env python3
"""
A first attempt to clean up image_analysis.py
Does not include preprocessing steps
"""

#Imports 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import time
import pickle
import tensorflow as tf 
import keras
from sklearn.preprocessing import StandardScaler


# %%
# Prepare the Dataset

resize_ratio = 1
sizex = round(480*resize_ratio)
sizey = round(640*resize_ratio)

bimgs = np.load('/data/lasso/sims/20160611/bimgs.npy')
whole_gradient = np.load('/data/lasso/sims/20160611/whole_gradient.npy')
front_gradient = np.load('/data/lasso/sims/20160611/front_gradient.npy')
rear_gradient = np.load('/data/lasso/sims/20160611/rear_gradient.npy')
middle_gradient = np.load('/data/lasso/sims/20160611/middle_gradient.npy')
average_near_pixel = np.load('/data/lasso/sims/20160611/average_near_pixel.npy')
img_save = np.load('/data/lasso/sims/20160611/img_save.npy')
length_of_cloud_ray = np.load('/data/lasso/sims/20160611/length_of_cloud_ray.npy')
fraction_inside_cloud = np.load('/data/lasso/sims/20160611/fraction_inside_cloud.npy')
length_inside_cloud = np.load('/data/lasso/sims/20160611/length_inside_cloud.npy')
label = np.load('/data/lasso/sims/20160611/label.npy')
label = label.astype(np.int)

number_of_files = label.shape[2]

Xcoords = np.arange(0,sizex)
Ycoords = np.arange(0,sizey)

# limit calculation to only contain the values inside the rendered region

XXC, YYC = np.meshgrid(Xcoords, Ycoords)
# D: Creates array with each pixel assigned radius from zenith
RRC = np.sqrt((XXC-sizex/2)**2+(YYC-sizey/2)**2)
# XXXC = np.ravel(XXC, order = 'C')
# YYYC = np.ravel(YYC, order = 'C')
RRRC = np.zeros((sizey,sizex, number_of_files))

for i in range(number_of_files):
    RRRC[:,:,i] = RRC

#shuffle images

np.random.seed(4097)
shuf = np.arange(number_of_files)
np.random.shuffle(shuf)

whole_gradient = whole_gradient[:,:,shuf]
front_gradient = front_gradient[:,:,shuf]
rear_gradient = rear_gradient[:,:,shuf]
middle_gradient = middle_gradient[:,:,shuf]
average_near_pixel = average_near_pixel[:,:,shuf]
bimgs = bimgs[:,:,shuf] # **4
rgb_r = img_save[:,:,0, :][:,:,shuf]
rgb_g = img_save[:,:,1, :][:,:,shuf]
rgb_b = img_save[:,:,2, :][:,:,shuf]
length_of_cloud_ray = length_of_cloud_ray[:,:,shuf]
fraction_inside_cloud = fraction_inside_cloud[:,:,shuf]
length_inside_cloud = length_inside_cloud[:,:,shuf]
#rfz : Radius From Zenith
radius_from_zenith = RRRC[:,:,shuf]
label = label[:,:,shuf]

#%%
#Make training data
n_train = 100
n_test = 20

wg = np.ravel(whole_gradient[:,:,0:n_train])
fg = np.ravel(front_gradient[:,:,0:n_train])
rg = np.ravel(rear_gradient[:,:,0:n_train])
mg = np.ravel(middle_gradient[:,:,0:n_train])
anp = np.ravel(average_near_pixel[:,:,0:n_train])
bi = np.ravel(bimgs[:,:,0:n_train])
rr = np.ravel(rgb_r[:,:,0:n_train])
gg = np.ravel(rgb_g[:,:,0:n_train])
bb = np.ravel(rgb_b[:,:,0:n_train])
lcr = np.ravel(length_of_cloud_ray[:,:,0:n_train])
fic = np.ravel(fraction_inside_cloud[:,:,0:n_train])
lic = np.ravel(length_inside_cloud[:,:,0:n_train])
rfz = np.ravel(radius_from_zenith[:,:,0:n_train])
lb = label[:,:,0:n_train]

data_train = {'wg':wg, 'fg':fg, 'rg':rg, 'mg':mg, 'anp':anp, 'bi':bi, 'rr':rr,
             'gg':gg, 'bb':bb, 'fic':fic, 'rfz':rfz}

df_train = pd.DataFrame(data_train)

wgt = np.ravel(whole_gradient[:,:,n_train + 1: n_train +1 + n_test ])
fgt = np.ravel(front_gradient[:,:,n_train + 1: n_train +1 + n_test ])
rgt = np.ravel(rear_gradient[:,:,n_train + 1: n_train +1 + n_test ])
mgt = np.ravel(middle_gradient[:,:,n_train + 1: n_train +1 + n_test ])
anpt = np.ravel(average_near_pixel[:,:,n_train + 1: n_train +1 + n_test ])
bit = np.ravel(bimgs[:,:,n_train + 1: n_train +1 + n_test ])
rrt = np.ravel(rgb_r[:,:,n_train + 1: n_train +1 + n_test ])
ggt = np.ravel(rgb_g[:,:,n_train + 1: n_train +1 + n_test ])
bbt = np.ravel(rgb_b[:,:,n_train + 1: n_train +1 + n_test ])
lcrt = np.ravel(length_of_cloud_ray[:,:,n_train + 1: n_train +1 + n_test ])
fict = np.ravel(fraction_inside_cloud[:,:,n_train + 1: n_train +1 + n_test ])
lict = np.ravel(length_inside_cloud[:,:,n_train + 1: n_train +1 + n_test ])
rfzt = np.ravel(radius_from_zenith[:,:,n_train + 1: n_train +1 + n_test ])
lbt = label[:,:,n_train + 1: n_train + 1 + n_test ]

data_test = {'wg':wgt, 'fg':fgt, 'rg':rgt, 'mg':mgt, 'anp':anpt, 'bi':bit, 'rr':rrt,
             'gg':ggt, 'bb':bbt, 'fic':fict, 'rfz':rfzt}

df_test = pd.DataFrame(data_test)

X_train = df_train.values
X_train = X_train.astype(np.float)
y_train = np.ravel(lb)

mask = (y_train > 1)
X_train = X_train[mask]
y_train = y_train[mask]
y_train = y_train % 2

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_test = df_test.values
X_test = X_test.astype(np.float)
y_test = np.ravel(lbt)

mask_test = (y_test > 1)
X_test = X_test[mask_test]
y_test = y_test[mask_test]
y_test = y_test % 2

X_test = scaler.transform(X_test)



# Want smaller sample size


# pca = PCA()

# pca.fit(X_train)

# # Plot the explained variances
# features = range(1, pca.n_components_ + 1)
# plt.bar(features, pca.explained_variance_ratio_)
# plt.xlabel('PCA feature')
# plt.ylabel('Explained variance')
# plt.xticks(features)
# plt.show()


# pca2 = PCA(n_components = 3)
# X_reduced = pca2.fit_transform(X_train)

# %% Check to make sure labels match up

rand_img_no = 222
rand_img = img_save[:,:,:,rand_img_no]
rand_lb = label[80:560,:,rand_img_no]
for i in range(480):
    for j in range(480):
        if np.sqrt((i-240)**2 + (j-240)**2) > 240:
            rand_lb[i,j] = np.nan
plt.imshow(rand_img/255)
plt.show()
plt.imshow(rand_lb)
plt.show()


# %%
# Initialize and train the model

strategy = tf.distribute.MirroredStrategy(devices=['/gpu:1'])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta = 1e-4,
                                           patience=5)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
class_weights_dict = dict(enumerate(class_weights))

with strategy.scope():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units = 11, input_shape = (11,), activation = 'relu'))
    model.add(tf.keras.layers.Dense(units = 11, activation = 'relu'))
    model.add(tf.keras.layers.Dense(units = 4, activation = 'relu'))
    model.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size = 1024, epochs = 100,
             callbacks = [callback], class_weight = class_weights_dict)

#%% Test the Model
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(-1)
y_pred_labels = np.round(y_pred)
y_pred_labels = y_pred_labels.astype(np.int)

from sklearn.metrics import confusion_matrix

def fcm(y_true, y_pred, labels = None, normalize = None):
    """
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by classifier
    
    labels: array-like of shape (n_classes), default=None
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels. 
        If None is given, those that appear at least once 
        in y_true or y_pred are used in sorted order. Also used
        to label the confusion matrix x and y tick labels

    normalize: {'true', 'pred', 'all}, default = None
        Normalizes confusion matrix over the true (rows),
        predicted(columns) or all the population. If None,
        confusion matrix will not be normalized

    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix whose i-th row and j-th column entry indicates 
        the number of samples with true label being i-th class 
        and prediced label being j-th class. 

    See documentation for sklearn.metrics.confusion_matrix and 
    https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
    """

    C = confusion_matrix(y_true, y_pred, labels, normalize)

    n = C.shape[0]
    fig, ax = plt.subplots()
    im = ax.imshow(C)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    for i in range(n):

        for j in range(n):

            ax.text(j, i, C[i, j],
                    ha = "center", va = "center",
                    color = "w")
    
    ax.set_title('Confusion Matrix')
    
    if labels != None:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=  45, ha = "right",
            rotation_mode = "anchor")

    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    fig.colorbar(im)
    fig.tight_layout()
    plt.show(block = True)
    plt.savefig(fname='confusionMatrix.png')

fcm(y_test, y_pred_labels)

from sklearn.metrics import f1_score

score2 = f1_score(y_test, y_pred_labels, pos_label = 0)
score3 = f1_score(y_test, y_pred_labels, pos_label = 1)

print('Cloud Base F1 score:' + str(score2))
print('Cloud Side F1 score:' + str(score3))


# %%
from sklearn.metrics import roc_curve, auc, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color = 'darkorange', 
         lw = 2, label = 'ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color = 'navy', lw = 2, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc = 'lower right')
plt.savefig(fname = 'rocCurve.png')
plt.show()

# %%
# EDA
y = np.ravel(lb)
mask_nonzero = (y != 0)
df_nonzero = df_train[mask_nonzero]

desc_df = pd.DataFrame(np.round(df_nonzero.describe(), 3))
print(desc_df.to_latex())

corrMatrix = np.round(df_nonzero.corr(), 2)

import seaborn as sn

sn.heatmap(corrMatrix, annot=True)
plt.tight_layout()
plt.show()

# %%
df_smaller = df_train.drop(columns = ['fg','rr','gg','bb'])
X_smaller = df_smaller.values
X_smaller = X_smaller.astype(np.float)
X_smaller = X_smaller[mask]

scaler = StandardScaler()
X_smaller = scaler.fit_transform(X_smaller)

dft_smaller = df_test.drop(columns = ['fg', 'rr', 'gg', 'bb'])
Xs_test = dft_smaller.values
Xs_test = Xs_test.astype(np.float)
Xs_test = Xs_test[mask_test]

Xs_test = scaler.transform(Xs_test)

strategy = tf.distribute.MirroredStrategy(devices=['/gpu:1'])
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta = 1e-4,
                                           patience=5)

with strategy.scope():
    model2 = tf.keras.Sequential()
    model2.add(tf.keras.layers.Dense(units = 7, input_shape = (7,), activation = 'relu'))
    model2.add(tf.keras.layers.Dense(units = 7, activation = 'relu'))
    model2.add(tf.keras.layers.Dense(units = 4, activation = 'relu'))
    model2.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
    model2.compile(optimizer = 'adam', loss = tf.keras.losses.BinaryCrossentropy(),
                 metrics=['accuracy'])

    model2.fit(X_smaller, y_train, batch_size = 1024, epochs = 100,
             callbacks = [callback])

# %%
