# datashell.py

`datashell.py` contains a class `DataShell` that acts as a container for TSI image data.

## class `DataShell`

### Attributes
---
- `fnames`: A list a filenames written as strings. Each file should contain a numpy array for a single variable and should be a of shape (image rows, image columns, number of samples). These arrays will most likely be outputs from `preprocessing.py`
- `img_fname`: A filename written as a string. The file itself is a numpy array containing the rgb values for all of the sample images. It should be of size (image rows, image columns, 3, number of samples)
- `lb_fname`: A filename written as a string. The file should be the labels that correspond to the images. It should be of size (image rows, image columns, number of samples)
- `directory`: A path written as a string. the path that the files above can be found in.

The following attributes are set as a result of using a class method:

- `xtrain`: The training data as a numpy array
- `xtest`: The test data as a numpy array
- `ytrain`: The training labels a numpy array
- `ytest`: The test labels a numpy array

### Methods
---

Once the initial attributes are set, you may call any of the following methods


- `load_data(fnames, directory)`: Loads a pre-existing data set from a list of .npy (or .npz) files. It expects the list to be ordered like ['training_set_filename', 'test_set_filename', 'training_labels_filename', 'test_labels_filename']. This method will populate the final attributes of the class instance.

- `train_test_split(num_train, num_test, seed = 1)`: Creates a shuffled training and test dataset (as numpy arrays) from the files specified in the attribute variables. A variable `radius` that measures the radius from the zenith is automatically created and added to the datasets. This array is the same for all images, and saving a number of copies equal to the sample size would be a waste of hard memory. This method will populate the final attributes of the class instance. 

- `to_csv(fnames, directory)`: Saves created datasets as CSV or text files, depending on extensions given in the list `fnames`. The order of the filenames should be the same as in `load_data`.

-  `to_npArray(fnames, directory)`: Saves created datasets as .npy (or .npz) files. The order of the filenames should be the same as in `load_data`

### Example Usage
---

<TODO>