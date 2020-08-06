import cupy as cp
import numpy as np
import netCDF4 as nc
import time

cp.cuda.Device(1).use()

# User Options
directory = '/data/lasso/sims/20160611/'
tloc = 30600
xloc = 6400


#Load data
fname = directory + 'ql.nc'
f = nc.Dataset(fname,'r')
x =  f.variables['x'][:]
y =  f.variables['y'][:]
z =  f.variables['z'][:]
time_array = f.variables['time'][:]

t = np.int0(np.squeeze(np.where(time_array == tloc)))

# Load liquid water content array
ql = cp.array(f.variables['ql'][t, :, :, :], dtype = cp.bool)

# Some useful variables
ni = 480
nj = 480
theta_max = 4/9 * np.pi # 80 degrees
ni0 = ni / 2 - 1
nj0 = nj / 2 - 1
jv, iv = np.meshgrid(np.arange(nj) - nj0, np.arange(ni) - ni0)

# Make arrays with the polar coordinates of the image
theta_img = theta_max / nj0 * np.sqrt(iv**2 + jv**2) 
phi_img = np.arctan2(iv, jv)

#icam will change in increments of 6 later on
icam = 256
jcam = np.int0(np.squeeze(np.where((x - 12.5) == xloc)))
xcam = x[jcam]
ycam = y[icam]

# Make arrays with spherical coordinates of cloud field
"""
shape (2nd entry, 1st entry, 3rd entry)
i.e. put entries in order y, z, x to align
precisely with ql
"""
yv, zv, xv = np.meshgrid(y - ycam, z, x - xcam)
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
labels = cp.zeros((ni, nj, 11))

# Doing all computations on GPU 1
with cp.cuda.Device(1):
    # Condition to search only in field of view
    limit_theta = thetav <= theta_max
    # Set closeness tolerance
    # Redefine useful variables so they are on the GPU
    # Reduces data transfer time
    ni = 480
    nj = 480
    theta_max = 4/9 * np.pi
    ni0 = ni / 2
    nj0 = nj / 2
    dimk, dimi, dimj = rv.shape

    # Start the loops!
    tic = time.time()
    for i in range(ni):
        for j in range(nj):
            
            # Only do work for pixels in inside the "circle"
            if cp.sqrt((i - ni0)**2 + (j - nj0)**2) <= nj0:

                # Set initial tolerance for the "closeness" of
                # theta and phi to theta_img[i,j] and phi[i,j] resp.
                # theta_tol = theta_tol_array[i,j]
                # phi_tol = phi_tol_array[i,j]
                tol = 8e-3

                #isolate ray along theta, phi
                theta_ray = cp.isclose(thetav, theta_img[i,j], atol=tol)
                select_thetas = cp.logical_and(limit_theta, theta_ray)
                phi_ray = cp.isclose(phiv, phi_img[i,j], atol=tol)
                select_ql = cp.logical_and(select_thetas, phi_ray)
                qlk, qli, qlj = cp.where(select_ql == True)

                #sort ray by radius
                radii = rv[qlk, qli, qlj]
                sort = cp.argsort(radii)
                qlk = qlk[sort]; qli = qli[sort]; qlj = qlj[sort]
                
                # Now shift points of view
                for num in cp.arange(0,11,1):
                    
                    #move camera position 6 entries forward in y-direction
                    qli += num

                    # make dem labels
                    try:
                        ql_ray = ql[qlk, qli, qlj]

                    # if any entry in qlk, qli, qlj outside respective
                    # array dimension sizes, get rid of those entries.
                    except:
                        getbad_k = qlk > dimk
                        getbad_i = qli > dimi
                        getbad_j = qlj > dimj
                        bad_ki = cp.logical_or(getbad_k, getbad_i)
                        bad_kij = cp.logical_or(bad_ki, getbad_j)
                        qlk = qlk[~bad_kij]
                        qli = qli[~bad_kij]
                        qlj = qlj[~bad_kij]

                        # Then make labels
                        ql_ray = ql[qlk, qli, qlj]
                        if ql_ray.size == 0:
                            labels[i,j,num] = cp.nan 

                        elif cp.all(~ql_ray) == True:
                            labels[i,j,num] = 1

                        else:
                            idx = cp.argmax(ql_ray)
                            if ql[qlk[idx]-1, qli[idx], qlj[idx]]:
                                labels[i,j,num] = 3

                            else:
                                labels[i,j,num] = 2

                    # If all entries in qlk, qlj, qli okay, proceed to labeling
                    else:
                        ql_ray = ql[qlk, qli, qlj]
                        if ql_ray.size == 0:
                            labels[i,j,num] = cp.nan 

                        elif cp.all(~ql_ray) == True:
                            labels[i,j,num] = 1

                        else:
                            idx = cp.argmax(ql_ray)
                            if ql[qlk[idx]-1, qli[idx], qlj[idx]]:
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
np.save(directory + 'labels_method2_' + str(tloc) + '_' + str(xloc) + '.npy', labels)