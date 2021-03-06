import os
import cv2
import time
import numpy as np
from scipy.ndimage import sobel

tloc = 30600
xloc = 9600

DIRECTORY = '/data/lasso/sims/20160611'
SUBDIRECTORY = '/newdata/' + str(tloc) + '_' + str(xloc) + '/'
INDIV_FRAMES = DIRECTORY + '/indiv_frames'
files = os.listdir(INDIV_FRAMES)

# Set seed for reproducibility
np.random.seed(seed = 101)

num_files = len(files)
mask = np.zeros(num_files)

# sort files into masked and not_masked
for i in range(num_files):
    if 'MASKED' in files[i]:
        mask[i] = 1
    else:
        mask[i] = 0

files_arr = np.array(files)
files_rend = np.sort(files_arr[mask == 0])

files_to_loop = []

for filename in files_rend:
    if str(tloc) in filename:
        if str(xloc) in filename:
            files_to_loop.append(filename)
        else:
            continue
    else:
        continue

number_files_to_loop = len(files_to_loop)

with open(DIRECTORY + SUBDIRECTORY + 'images_used.txt', 'w') as filehandle:
    for filename in files_to_loop:
        filehandle.write("%s\n" % filename)

sizex = 480
sizey = 640

x0 = sizex / 2 - 0.5
y0 = sizey / 2 - 0.5
x = np.arange(sizex) - x0
y = np.arange(sizey) - y0

X, Y = np.meshgrid(x, y)
R = np.sqrt(X ** 2 + Y ** 2)

# Initial dataset feature arrays

# brightness of the image (the 2 norm of the RGB values)
bright = np.zeros((sizey, sizex, number_files_to_loop))
# gradient of the brightness
x_grad = np.zeros((sizey, sizex, number_files_to_loop))
y_grad = np.zeros((sizey, sizex, number_files_to_loop))
# RGB values
r = np.zeros((sizey, sizex, number_files_to_loop))
g = np.zeros((sizey, sizex, number_files_to_loop))
b = np.zeros((sizey, sizex, number_files_to_loop))
# Sobel filter on RGB values
r_sobel = np.zeros((sizey, sizex, number_files_to_loop))
g_sobel = np.zeros((sizey, sizex, number_files_to_loop))
b_sobel = np.zeros((sizey, sizex, number_files_to_loop))
# radius from zenith
radius = np.zeros((sizey, sizex, number_files_to_loop))
# array to save images in
img_array = np.zeros((sizey, sizex, 3, number_files_to_loop))

step = np.int0(number_files_to_loop / 100)
give_upates = np.arange(step, number_files_to_loop + step, step)

for l in range(number_files_to_loop):

    img = cv2.imread(INDIV_FRAMES + '/' + files_to_loop[l])
    img_array[:,:,:,l] = img
    bright[:,:,l] = np.sqrt(np.sum(img ** 2, axis = 2))
    y_grad, x_grad = np.gradient(bright[:,:,l])
    r[:,:,l], g[:,:,l], b[:,:,l] = cv2.split(img)
    radius[:,:,l] = R
    sobel_y = sobel(img, axis = 0)
    sobel_x = sobel(img, axis = 1)
    sobel_ = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # clean up sobel filter
    for i in range(sizey):
        for j in range(sizex):
            for k in range(3):
                sobel_[i,j,k] = np.min([255, sobel_[i,j,k]])

    r_sobel[:,:,l] = sobel_[:,:,0]
    g_sobel[:,:,l] = sobel_[:,:,1]
    b_sobel[:,:,l] = sobel_[:,:,2]

    if l in give_upates:
        p = l / number_files_to_loop * 100
        print("%0.2f percent complete" % p)



# Save all arrays in a new directory
np.save(DIRECTORY + SUBDIRECTORY + 'bright.npy', bright)
np.save(DIRECTORY + SUBDIRECTORY + 'x_grad.npy', x_grad)
np.save(DIRECTORY + SUBDIRECTORY + 'y_grad.npy', y_grad)
np.save(DIRECTORY + SUBDIRECTORY + 'r.npy', r)
np.save(DIRECTORY + SUBDIRECTORY + 'g.npy', g)
np.save(DIRECTORY + SUBDIRECTORY + 'b.npy', b)
np.save(DIRECTORY + SUBDIRECTORY + 'r_sobel.npy', r_sobel)
np.save(DIRECTORY + SUBDIRECTORY + 'g_sobel.npy', g_sobel)
np.save(DIRECTORY + SUBDIRECTORY + 'b_sobel.npy', b_sobel)
np.save(DIRECTORY + SUBDIRECTORY + 'radius.npy', radius)
np.save(DIRECTORY + SUBDIRECTORY + 'img_array.npy', img_array)



