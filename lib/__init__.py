import scipy.io

def get_usable_datasets(fname):    
    mat = scipy.io.loadmat(fname)
    usable = [x for x in mat['use'][0]] # file listing
    usable = [x[0] for x in usable] # fnames only
    usable = [x.split(".")[0] for x in usable] # dataset ids only
    return [x for x in usable]