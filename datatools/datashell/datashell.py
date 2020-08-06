import numpy as np 

class ValueTooLarge(Exception):
    pass

def dict_to_array(X):
    ''' Convert dictionary of data into numpy array'''

    xlist = list(X.values())
    # transpose so entries so that each entry in 
    # dictionary is treated like a feature
    xarray = np.array(xlist)
    return xarray.T


def make_data(fnames, lb_fname, directory, seed):
    ''' Puts data into dictionary'''

    np.random.seed(seed)
    # load labels, recast as integers
    labels = np.load(directory + lb_fname)
    labels = labels.astype(np.int)

    # make array to shuffle data with
    num_files = labels.shape[2]
    shuf = np.arange(num_files)
    np.random.shuffle(shuf)

    # unroll and shuffle labels
    labels = labels[:,:,shuf]

    data_dict = {}

    # loop through filenames and load each array in,
    # then shuffle it and add to dictionary
    for file in fnames:
        arr = np.load(directory + file)
        data_dict[file] = arr[:,:,shuf]

    return data_dict, labels


def split_data(X, y, num_train, num_test = None):

    sample_size = y.shape[2]

    if num_train > sample_size:
        raise ValueTooLarge("num_train exceeds dataset size")

    else:
        if num_test == None:
            num_test = sample_size - num_train

        elif num_test + num_train > sample_size:
            raise ValueTooLarge("num_train + num_test exceeds dataset size")

    xtrain = {}
    xtest = {}

    for arr in X.keys():
        get_train = X[arr][:,:,0:num_train]
        xtrain[arr] = np.ravel(get_train)
        get_test = X[arr][:,:,num_train:num_train + num_test]
        xtest[arr] = np.ravel(get_test)

    xtrain = dict_to_array(xtrain)
    xtest = dict_to_array(xtest)

    get_ytrain = y[:,:,num_train]
    ytrain = np.ravel(get_ytrain)
    get_ytest = y[:,:,num_train:num_train + num_test]
    ytest = np.ravel(get_ytest)

    return xtrain, xtest, ytrain, ytest


class DataShell:
    ''' A data container for TSI image data '''

    def __init__(self, fnames, lb_fname, directory):
        self.fnames  = fnames
        self.lb_fname = lb_fname
        self.directory = directory
        self.xtrain = None
        self.xtest = None
        self.ytrain = None
        self.ytest = None
        

    def load_data(self, fnames, directory = None):
        ''' Loads existing data from a list a filnames'''

        if directory == None:
            directory == self.directory

        self.xtrain = np.load(directory + fnames[0])
        self.xtest = np.load(directory + fnames[1])
        self.ytrain = np.load(directory + fnames[2])
        self.ytest = np.load(directory + fnames[3])


    def train_test_split(self, num_train, num_test = None, seed = 1):
        ''' Reads in data and makes train and test sets '''

        X, y = make_data(self.fnames, self.lb_fname, self.directory, seed)
        self.xtrain, self.xtest, self.ytrain, self.test = split_data(X, y, num_train, num_test)

    def to_csv(self, fnames, directory = None):
        ''' Saves train and test sets as .csv files '''
        if directory == None:
            directory = self.directory

        np.savetxt(directory + fnames[0], self.xtrain, delimiter=',')
        np.savetxt(directory + fnames[1], self.xtest, delimiter=',')
        np.savetxt(directory + fnames[2], self.ytrain, delimiter=',')
        np.savetxt(directory + fnames[3], self.ytest, delimiter=',')

    def to_npArray(self, fnames, directory = None):
        ''' Saves train and test sets as .npy files '''

        if directory == None:
            directory = self.directory

        np.save(directory + fnames[0], self.xtrain)
        np.save(directory + fnames[1], self.xtest)
        np.save(directory + fnames[2], self.ytrain)
        np.save(directory + fnames[3], self.ytest)
    