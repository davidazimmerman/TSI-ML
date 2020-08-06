import time
import cupy as cp
import numpy as np
import netCDF4 as nc 


DIRECTORY = '/data/lasso/sims/20160611/'
tloc = 30600

filename = DIRECTORY + 'ql.nc'
f = nc.Dataset(filename, 'r')
x = np.array(f.variables['x'][:])
y =  np.array(f.variables['y'][:])
z =  np.array(f.variables['z'][:])
time_array = f.variables['time'][:]

t = np.int0(np.squeeze(np.where(time_array == tloc)))

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

"""
shape (2nd entry, 1st entry, 3rd entry)
i.e. put entries in order y, z, x to align
precisely with ql
"""
yv, zv, xv = np.meshgrid(y - 6400, z, x - 6400)
rv = np.sqrt(xv**2 + yv**2 + zv**2)
thetav = np.arccos(zv / rv)
phiv = np.arctan2(yv, xv)
del xv, yv, zv, jv, iv

cp.cuda.Device(1).use()

# Load arrays onto GPU using CuPy module
# theta_tol_array = cp.array(theta_tol_array)
# phi_tol_array = cp.array(phi_tol_array)
theta_img = cp.array(theta_img)
phi_img = cp.array(phi_img)
thetav = cp.array(thetav)
phiv = cp.array(phiv)
rv = cp.array(rv)

xcoords = [6400, 9600, 12800, 16000, 19200]

# Calculate boundary layer
bl = []
for k in range(ql.shape[0]):
    ql_slice = ql[k,:,:]
    if np.all(~ql_slice) == False:
        bl.append(k)
    else:
        continue

start = min(bl)
stop = max(bl)

with cp.cuda.Device(1):

    ni = 480
    nj = 480
    theta_max = 4/9 * cp.pi # 80 degrees
    ni0 = ni / 2 - 0.5
    nj0 = nj / 2 - 0.5
    dimk, dimi, dimj = rv.shape
    boundary = cp.array([0, 1023])

    map_array = cp.zeros((480, 480, stop - start, 3), dtype=cp.int)

    tic = time.time()
    for i in range(ni):
        for j in range(nj):

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

                map_array[i,j,0:how_many,0] = kx
                map_array[i,j,0:how_many,1] = ix
                map_array[i,j,0:how_many,2] = jx

                map_array[i,j,how_many:,:] = cp.nan
            
            else:
                map_array[i,j,:,:] = cp.nan

np.save(DIRECTORY + 'map_array.npy', map_array)

