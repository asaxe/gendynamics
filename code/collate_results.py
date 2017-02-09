import numpy as np
import glob


expt = 1

files = glob.glob("../results/expt%d/*.npz" % expt)

train = np.empty((5000,))
test = np.empty((5000,))
exacttest = np.empty((5000,))

def load_file(fn,idx):
    res = np.load(fn)
    if len(res.files) > 1:
        return (res[idx])
    else:
        return ()
    
def load_params(fn):
    res = np.load(fn)
    if len(res.files) > 1:
        t = np.array([res['params']])
        return  t[0]
    else:
        return ()

train = np.vstack([load_file(f,'train') for f in files])
test = np.vstack([load_file(f,'test') for f in files])
exact_test = np.vstack([load_file(f,'exact_test') for f in files])
params = np.vstack([load_params(f) for f in files])[:,0]

