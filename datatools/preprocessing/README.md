# Data Preprocessing and Feature Engineering

## `preprocessing.py`

This script is essentially only the steps used to create the dataset used in Nick Barron's original `image_analysis.py` script, with the directories changed to be valid directories on the Cumulus server. The one notable difference is that, for whatever reason, when you the masked and rendered lists of filenames are created, they are not in the same order! To correct this, the lists are first sorted, and then randomly (but identically!) shuffled, so that we have a dataset of random samples. The name of the files used are saved in a .txt file in the same directory as the rest of the simulation data. Otherwise, some comments have been added, but this was mostly C&P'd directly from Nick's original implementation.

Labels are also created while running, using the K-means clustering method used in `label_maker_method_1.py` in the `lbtools` subdirectory. As mentioned in the documentation for that directory, this method is not really reliable. Moreover, many of the features created for the dataset use the label itself. Furthermore, the resulting dataset is extremely sparse, and also suffers greatly from multicollinearity. With an improved labeling method, this issue may or may not resolve itself, and some of the features may or may not be important predictors.

## `newdata.py`

This script creates a different dataset, with features that are not dependent on the labels. Rather than take a random sample of images, images from a specific time and camera position (x-axis) are chosen, with the idea that once enough datasets like this have been made, you can take a random sample from those. The names of the files used are saved in a .txt file in the same subdirectory as the resulting dataset, although this is probably not necessary, since the files choosen correspond to all images from the choose time and location. The script also saves the images used in a numpy array, for easy loading.

### Feature Engineering

The features for this dataset are described here:

- `r`, `g`, `b`: The rgb values of the image itself
- `bright`: The brightness of the image, as measured by the 2-norm of the rgb values
- `x_grad`, `y_grad`: The gradient of the brightness, computing using the 2nd order central difference method. See documentation for the [numpy gradient function](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html), but check which version of numpy Cumulus is using.
- `r_sobel`, `g_sobel`, `b_sobel`: The [sobel filter](https://en.wikipedia.org/wiki/Sobel_operator) applied to the RGB values, with help from the [sobel function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html) from scipy.ndimage
- `radius`:  The radius of the pixel from the zenith (camera position)

The gradient and sobel features are especially important, because they contain information about surrounding pixels. However, only pixels directly adjacent are considered. Nick's idea with some of his features was to consider information from pixels even futher away, and I think this is probably wise and would improve prediction confidence and power. 