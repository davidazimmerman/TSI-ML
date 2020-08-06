import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

def kmeans_labels(directory, landingname, tloc, xloc):
    """
    Makes labels for TSI image data using Kmeans clustering

    Paramters
    ---------
    xloc: int
        camera x-position. Must be 6400, 9600, 12800, 16000, or 19200
    tloc: int
        time. Must be one of the actual times used
    directory: string
        path where MASKED .png files can be found
    landingname: string
        path to save labels

    Returns
    -------
    Saves an array of size (640, 480, 111) where the third axis/index is 
    for the image number. This file is saved in the directory 'landingname' as 
    'labels_kmeans_<tloc>_<xloc>.npy'
    """

    tloc_str = str(tloc)
    xloc_str = str(xloc)

    files = os.listdir(directory)
    mask = np.zeros(len(files))
    for i in range(len(files)):
        if 'MASKED' in files[i]:
            mask[i] = 2
        else:
            mask[i] = 1

    files = np.array(files)     
    files_mask = files[mask == 2]
    files_rend = files[mask == 1]

    selection = []

    for i in range(files_mask.size):
        if tloc_str in files_mask[i]:
            if xloc_str in files_mask[i]:
                selection.append(files_mask[i])
            else:
                continue
        else:
            continue

    labels = np.zeros((640, 480, len(selection)))
    init = np.array([[0, 0, 0], [7, 228, 95],
                    [105, 124, 109], [190, 205, 193]])/255

    for i in range(len(selection)):
        # get image and make dataset
        img = cv2.imread(directory + '/' + selection[i])
        r, g, b = cv2.split(img)
        X = np.array([r.flatten(), g.flatten(), b.flatten()], dtype = np.float).T
        X /= 255

        # perform clustering
        kmeans = KMeans(n_clusters = 4, random_state = 1,
                        init = init, n_init = 1)

        Y = kmeans.fit_predict(X)
        labels[:,:,i] = np.reshape(Y, (640, 480))

    np.save(landingname + '/labels_kmeans_' + tloc_str + '_' + xloc_str + '.npy', labels)

        


    





