import numpy as np

def make_bg_cuts(cuts,Np):
    cuts = [1] + cuts + [1e99]
    bg_cuts = np.zeros(Np)
    cidx = 0
    for i in range(1,Np+1):
        #print(i)
        if (cuts[cidx] <= i < cuts[cidx+1]):
            bg_cuts[i-1] = cuts[cidx]
        else:
            cidx += 1
            bg_cuts[i-1] = cuts[cidx]
            
    Ncuts = len(np.unique(bg_cuts))
    idx_arr = np.ones(Np)
    for i in range(Ncuts):
        idx_arr[np.where(bg_cuts == cuts[i+1])[0]] = i+1
    
    return bg_cuts.astype(int), idx_arr.astype(int), Ncuts


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
