# Label Making

Each script in this directory is used to make labels for the image data. Each pixel receives a label of 0 (null), 1 (open sky), 2 (cloud base), or 3 (cloud side). There are some parts of clouds that may be both side and base, meaning that the area to the side of the region under consideration is open sky, and the area below it is also open sky. In this case, the region is classified as base. It may be interesting to add a new category of side-sky and see if this improves model performance. Each of the scripts uses a different method for doing this, although methods 1 and 2 are similar in spirit. Method 1 is the method that I inherited from the previous project owner. 

## `label_maker_method_1.py`

This script uses K-means clustering on the image files that were supposed to be the labeled output from blender, the rendering software used to make the simulations. However, these are 24 bit .png files (each pixel still has an RGB value), and there are up to 8K unique RGB values in a single image. To consildate these RGB values into four classes, K-means clustering looks for natural clusters among these RGB values, and assigns each pixel a label. Unfortunately, the original blender output is pretty mysterious, and nobody really understands how it was generated. Therefore these labels are unreliable. 

Furthermore, the original dataset includes features that were generated using these labels (see `image_analysis.py` and `preprocessing.py`), and so many of these features are meaningless. While training with this dataset and the corresponding labels, I never achieved better than "slightly better than random guessing" when predicting cloud side, which is the rarest class. 

This script is saved like a module and can be imported into a script or interactive environment. It takes less than a couple of minutes to run on Cumulus.

### function `kmeans_labels`

#### Parameters
---
- `directory`: string
    directory where the 'MASKED' files may be found
- `landingname`: string
    directory in which the labels are to be saved
- `tloc`: int
    time value for the images
- `xloc`: int
    x-coordinate of camera. Must be 6400, 9600, 12800, 16000, or 19200

#### Returns
---
Saves an array of size (640, 480, 111) where the third axis/index is 
for the image number. This file is saved in the directory 'landingname' as 
'labels_kmeans_`<tloc>`_`<xloc>`.npy'

## `label_maker_method_2.py`

This script takes a very different approach to creating labels for the data. It actually looks at some of the data from the simulation itself, which is stored in the hdf5 file `ql.nc`. One of the arrays in this file is `ql`, which is four dimensional with axes (time, z(height from ground), y, x) and contains the liquid water content at each position. The variable `tloc` is the time index for this array, and `xloc` is the x position of the camera. The field of view of the camera is $80^{\circ}$ from the zenith. `t` is the index for time = `tloc`. The array `ql[t,:,:,:]` is converted to a boolean array, since we only need to know if liquid water is present to determine if that region contains a cloudy object. 

Three coordinate arrays `rv`, `phiv` and `thetav` are created. These are the spherical coordinates of `ql[t,:,:,:]`. Then `phi_img` and `theta_img` are created, and are the "polar" coordinates in the image. Since the camera lens is spherical, `theta_img` is really just the radius from the zenith scaled so that it's max is $80^{\circ}$ ($\frac{4}{9}\pi$ radians). The script loops through each pixel in an image, and finds all entries in `ql[t,:,:,:]` that have equal $(\theta, \phi)$ coordinates within some tolerance and looks for cloudy objects. Choosing this tolerance has been problematic, as a small tolerance will select too many values in `ql`, thus increasing the chance of mislabeling, and seems to result in making images that look like blobs. A tolerance that is too small has too few values in `ql`, and may fail to detect cloudy objects, or may even just result in that pixel being labeled a null value. 

The script is a not saved like a module, and needs to be modified before each use and run from the command line. It is probably best to run it using `nohup`, as it takes about 4 hours to run on Cumulus. The variables that need to be changed are near the top under the comment 'User Options'. Currently, the script assumes that you are saving the labels in the same directory as `ql`. This can easily be adjusted

## `label_maker_method_3.py.bkp`

This script takes a similar approach to creating labels as `label_maker_method_2.py`, however, the problem of choosing appropriate tolerances is eliminated. The script will search through the boundary layer, layer by layer, and find the entry in the layer that is closest to $(\theta, \phi)$ and add this entry to an array. Then we can guarantee that only one entry from every z coordinate gets selected for our "ray". Then, as before, a search for cloudy objects is conducted. This is hands down the slowest of the 3 methods, but seems to be the most accurate. That being said, near the edges of the images there seems to be some "smearing" of cloud base. I am not sure why this is happening but is definitely something to look into. 

Similar to the previous method, this script is not saved like a module and needs to be modified in the same way. Definitely use `nohup` when running from the command line as it takes about 13 hours to run. 

## `label_maker_method_3.py.bkp`

This script is an updated version of `label_maker_method_3.py.bkp` that aims to fix the smearing issue and also resolves some other issues that only recently came to my attention. The ray that was being used to find cloudy objects was being made after the shift in y-coordinates. I don't think this was the right thing to do, and have adjusted so this doesn't happen. This may also have the added benefit of cutting some computation time.

Also, `theta_img` it seems has not been computed correctly in previous iterations of this method (or any other). The field of view is $80^{\circ}$, and I have previously assumed that the outer layer was no more than 240 units away from the camera position, but there are rgb values in images that are nonzero (nonzero vector, that is) for pixels that are 241.463... units out. This number was discovered by finding all pixels in a random image that were "edge" pixels, meaning that they were othogonally neighboring pixels with (0,0,0) rgb value, and computed their radii from the camera position (center of the image), and taking the maximum. If you look closely, you can see that the images are actually sort of squashed at the boundary. 

## `generate_map.py`

This is an implementation of an earlier idea from Thijs. Ultimately, what we are doing in the label making scripts is making a map from a discrete 280 x 1024 x 1024 3D space to a discrete 480 x 480 (or 640 x 480) 2D space. Once we know what coordinates in the 3D space get mapped to what pixels in the 2D space for in the inital camera position, then we can simply adjust the y coordinates to adjust for each frame (for a fixed x coordinate. In fact, this is exactly what the `label_maker_method_2.py` and `label_maker_method_3.py` do) and then adjust the x coodinates for each camera to make labels for all frames given by all cameras for a fixed time. 

Since this map is the same for all time, it can be used to simultaneously make labels for all of the times. Thus we can use this idea to make all of the labels for an entire simulation. This script, as written creates a map array for the first camera position and saves it in some directory. It is not looping over the times _or_ camera positions (or even making labels!) but I intend to do this as eventually. This is just a proof of concept for the coordinate map.

## What is left to do?

Even with the improvements made, none of the labels created thus far are aligned properly with the images. It seems that the perspective is not quite right. Here are some more detailed descriptions of the problems I have encountered:

1. Supposedly, each image made by the simulation has the y-coordinate of the camera shifted by 6 units (150 meters), with the starting place being 256 units (6400 meters). However, typically, the first label generated by methods 2 or 3 corresponds to about the 11th rendered image. Furthermore, the subsequent labels do not relate linearly to the image numbers. For example, the 2nd label may correspond to image number 15 (not 12 as one would expect) and the 10th label may correspond to image number 68. This is pretty strange, and does not seem to be very consistent across simulations. This could be an issue with where we choose the center to be in the images (currently, the center is chosen to be (row = 320, col = 240)), but could also be an issue with how the spherical coordinates are defined. 

2. The edges of the labels created using method 3 having a smearing effect. I honestly have no idea what is causing this. Fortunately, it doesn't seem to be happening on the interior of the image, so if the problem above can be solved, it is possible that these labels may still be useful. ___EDIT:___ This issue is potentially fixed with updated version of method 3.