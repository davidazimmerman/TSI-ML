import cupy as cp
import numpy as np
import netCDF4 as nc
import time

cp.cuda.Device(1).use()

# User Options
directory = '/data/lasso/sims/20160611/'
indiv_frames = 'indiv_frames'
tloc = 30600
xloc = 9600

# Load data
fname = directory + 'ql.nc'
f = nc.Dataset(fname,'r')
x =  np.array(f.variables['x'][:], dtype=np.float64)
y =  np.array(f.variables['y'][:], dtype=np.float64)
z =  np.array(f.variables['z'][:], dtype=np.float64)
time_array = f.variables['time'][:]

t = np.int0(np.squeeze(np.where(time_array == tloc)))

# Load liquid water content array
ql = cp.array(f.variables['ql'][t, :, :, :], dtype = cp.bool)

# Some useful variables
ni = 480
nj = 480
theta_max = 4/9 * np.pi # 80 degrees
ni0 = ni / 2 - 0.5
nj0 = nj / 2 - 0.5
jv, iv = np.meshgrid(np.arange(nj) - nj0, np.arange(ni) - ni0)

# Make arrays with the polar coordinates of the image
theta_img = theta_max * np.sqrt(iv**2 + jv**2) / 241.4632477210559
phi_img = np.arctan2(iv, jv)

ycam = 6400
"""
shape (2nd entry, 1st entry, 3rd entry)
i.e. put entries in order y, z, x to align
precisely with ql
"""
yv, zv, xv = np.meshgrid(y - ycam, z, x - xloc)
rv = np.sqrt(xv**2 + yv**2 + zv**2)
thetav = np.arccos(zv / rv)
phiv = np.arctan2(yv, xv)
del xv, yv, zv, jv, iv



# Load arrays onto GPU using CuPy module
# theta_tol_array = cp.array(theta_tol_array)
# phi_tol_array = cp.array(phi_tol_array)
theta_img = cp.array(theta_img)
phi_img = cp.array(phi_img)
thetav = cp.array(thetav)
phiv = cp.array(phiv)
rv = cp.array(rv)

# Initialize label array
labels = cp.zeros((ni, nj, 111))

# Doing all computations on GPU 1
with cp.cuda.Device(1):
    # Condition to search only in field of view
    #limit_theta = thetav <= theta_max
    # Set closeness tolerance
    # Redefine useful variables so they are on the GPU
    # Reduces data transfer time
    ni = 480
    nj = 480
    theta_max = 4 / 9 * cp.pi
    ni0 = ni / 2 - 0.5
    nj0 = nj / 2 - 0.5
    dimk, dimi, dimj = rv.shape
    start = 76
    stop = 146
    boundary = cp.array([0, 1023])

    # Start the loops!
    tic = time.time()
    for i in range(ni):
        for j in range(nj):
            
            # Only do work for pixels in inside the "circle"
            if theta_img[i,j] <= theta_max:

                count = 0
                ray = cp.zeros((stop + 1 - start, 3), dtype=cp.int)

                for k in range(start, stop + 1):

                    if count > 0:
                        ray[k-start,:] = cp.array([cp.nan, cp.nan, cp.nan])
                    else:
                        theta_slice = thetav[k,:,:]
                        theta_slice[theta_slice > theta_max] = cp.nan
                        phi_slice = phiv[k,:,:]
                        dist = cp.sqrt((theta_slice - theta_img[i,j])**2 + (phi_slice - phi_img[i,j])**2)
                        mini, minj = cp.unravel_index(cp.nanargmin(dist), theta_slice.shape)
                        mini = cp.int(mini); minj = cp.int(minj)
                        # prevent smearing
                        if (mini in boundary) or (minj in boundary):
                            count += 1
                            
                        ray[k-start,0] = k
                        ray[k-start,1] = mini
                        ray[k-start,2] = minj

                not_nans = ~cp.isnan(ray[:,0])
                how_many = cp.sum(not_nans)
                kx = ray[:,0]
                kx = kx[not_nans]
                ix = ray[:,1]
                ix = ix[not_nans]
                jx = ray[:,2]
                jx = jx[not_nans]

                radii = rv[kx, ix, jx]
                sort = cp.argsort(radii)
                kx = kx[sort]
                ix = ix[sort]
                jx = jx[sort]

                
                # Now shift points of view
                for num in cp.arange(111):
                    
                    #move camera position 6 entries forward in y-direction
                    
                    ix += 6 * num

                    # make dem labels
                    try:
                        ql_ray = ql[kx, ix, jx]

                    # if any entry in qlk, qli, qlj outside respective
                    # array dimension sizes, get rid of those entries.
                    except:
                        getbad_k = kx > dimk
                        getbad_i = ix > dimi
                        getbad_j = jx > dimj
                        bad_ki = cp.logical_or(getbad_k, getbad_i)
                        bad_kij = cp.logical_or(bad_ki, getbad_j)
                        kx = kx[~bad_kij]
                        ix = ix[~bad_kij]
                        jx = jx[~bad_kij]

                        ql_ray = ql[kx, ix, jx]
                        # Then make labels

                        if cp.all(~ql_ray) == True:
                            labels[i,j,num] = 1

                        else:
                            idx = cp.argmax(ql_ray)

                            if ql[kx[idx]-1, ix[idx], jx[idx]] == True:
                                labels[i,j,num] = 3

                            else:
                                labels[i,j,num] = 2

                    # If all entries in qlk, qlj, qli okay, proceed to labeling
                    else:
                        ql_ray = ql[kx, ix, jx]

                        if cp.all(~ql_ray) == True:
                            labels[i,j,num] = 1

                        else:
                            idx = cp.argmax(ql_ray)

                            if ql[kx[idx]-1, ix[idx], jx[idx]]:
                                labels[i,j,num] = 3

                            else:
                                labels[i,j,num] = 2

            else:
                labels[i,j,:] = cp.nan
            

            # Give a bunch of updates every 5%. Because I want them, that's why.
            current_row = i + 1
            current_col = j + 1
            if ((current_row - 1) * 480 + current_col) % 11520 == 0:
                toc = time.time()
                time_elapsed = toc - tic
                p_complete = ((current_row - 1) * 480 + current_col)/ 480 ** 2
                print('-------------------------------------------------')
                print('PROCESS UPDATE')
                print('labels are %0.2f percent complete' %(p_complete * 100))
                print('current time elapsed: %0.2f minutes' %(time_elapsed / 60))
                print('-------------------------------------------------')

# Send labels back to CPU so they can be saved in hard memory
labels = cp.asnumpy(labels)
np.save(directory + 'labels_method3_' + str(tloc) + '_' + str(xloc) + '.npy', labels)