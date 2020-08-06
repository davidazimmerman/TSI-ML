#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.cluster import KMeans
import time

# %%
"""
C&P'd from Nick Barron's image_analysis.py. Masked files are 
ground truth values 
"""
indiv_frames = '/data/lasso/sims/20160611/indiv_frames/'
files = os.listdir(indiv_frames)
# split up the files into masked and rendered

mask = np.zeros(len(files))
for i in range(len(files)):
    if 'MASKED' in files[i]:
        mask[i] = 2
    else:
        mask[i] = 1
   
files = np.array(files)     
files_mask = files[mask == 2]
files_rend = files[mask == 1]

#%%

# Select a random image
img_label = cv2.imread(indiv_frames + files_mask[101])
img = cv2.imread(indiv_frames + files_rend[101])

r, g, b = cv2.split(img_label)
X = np.array([r.flatten(), g.flatten(), b.flatten()]).T

K = 10
inertia = np.zeros(K)

tic = time.time()
for k in range(K):

    kmeans = KMeans(n_clusters = k + 1, random_state = 1)

    kmeans.fit(X)

    inertia[k] = kmeans.inertia_

    if k == 3:
        print(kmeans.cluster_centers_)
toc = time.time()
print('run time for 10 kmeans on single image ' + str(toc - tic))

plt.plot(range(1, K + 1), inertia)
plt.arrow(
        x = 5, y = 2e9, dx = -0.8, dy = -1.5e9, 
        linewidth = 1, head_width = 0.3, head_length = 0.3e9
        )
plt.annotate(s = 'elbow', xy = (5,2e9))
plt.title('Training Cost vs. Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia (Training cost)')
plt.grid(True)
plt.savefig(fname='kmeansAnalysis.png')
plt.show()

# %%

