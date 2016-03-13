from numpy import dot, divide, int32, rint
import numpy as np
import multiprocessing as mp
from fdkload import fdkload
import pastset as ps
cpus = 2


def loadData(nr_projections,detector_rows, detector_columns, voxels):
    global pool
    global projections
    global combined_matrix
    global z_voxel_coords
    global transform_matrix
    global volume_weight
    global recon_volume
    pool =  mp.Pool(2)
    (projections, combined_matrix, z_voxel_coords, transform_matrix, volume_weight) = \
        fdkload(nr_projections, detector_rows, detector_columns, voxels)
    recon_volume = np.zeros((voxels,voxels,voxels))

def fdkworker(p,z,z_voxels,detector_rows, detector_columns, recon_volume,flat_proj_data,lock1,lock2):
    flat_proj_data = projections[p].ravel()
    # Put current z voxel into combined_matrix
    lock1.acquire()
    combined_matrix[2, :] = z_voxel_coords[z]
    lock1.release()

    # Find the mapping between volume voxels and detector pixels
    # for the current angle

    vol_det_map = dot(transform_matrix[p], combined_matrix)
    map_cols = rint(divide(vol_det_map[0, :], vol_det_map[2, :
                    ])).astype(int32)
    map_rows = rint(divide(vol_det_map[1, :], vol_det_map[2, :
                    ])).astype(int32)

    # Find the detector pixels that contribute to the current slice
    # xrays that hit outside the detector area are masked out

    mask = (map_cols >= 0) & (map_rows >= 0) & (map_cols < detector_columns) & (map_rows < detector_rows)

    # The projection pixels that contribute to the current slice

    proj_indexs = map_cols * mask + map_rows * mask \
        * detector_columns

    # Add the weighted projection pixel values to their
    # corresponding voxels in the z slice
    lock2.acquire()
    recon_volume[z].flat += flat_proj_data[proj_indexs] * volume_weight[p] * mask
    lock2.release()

def fdkouterpset(pset,projrange, z_voxels, detector_rows, detector_columns):
    global projections
    global combined_matrix
    global z_voxel_coords
    global transform_matrix
    global volume_weight
    global recon_volume
    recon_volume = np.zeros((z_voxels,z_voxels,z_voxels))

    #Get pset data!





    fdkcorehandler(projrange,projections,combined_matrix, z_voxel_coords,
            transform_matrix, z_voxels, detector_rows, detector_columns,
            recon_volume, volume_weight)

    return 0

def fdkouterstd(nr_projections, projections_, combined_matrix_, z_voxel_coords_,
            transform_matrix_, z_voxels, detector_rows, detector_columns,
            recon_volume_, volume_weight_):
    global projections
    global combined_matrix
    global z_voxel_coords
    global transform_matrix
    global volume_weight
    global recon_volume

    projections = projections_
    combined_matrix = combined_matrix_
    z_voxel_coords= z_voxel_coords_
    transform_matrix = transform_matrix_
    volume_weight= volume_weight_
    recon_volume = recon_volume_

    pool = mp.Pool(cpus)

    recon_volume = np.zeros((z_voxels,z_voxels,z_voxels))
    fdkcorehandler(xrange(nr_projections),projections,combined_matrix, z_voxel_coords,
            transform_matrix, z_voxels, detector_rows, detector_columns,
            recon_volume, volume_weight,pool)


    return recon_volume



def fdkcorehandler(projrange, projections, combined_matrix, z_voxel_coords,
            transform_matrix, z_voxels, detector_rows, detector_columns,
            recon_volume, volume_weight,pool):
    lock1 = mp.Lock()
    lock2 = mp.Lock()
    for p in projrange:
        # Numpy FDK operates on flat arrays
        flat_proj_data = projections[p].ravel()
        args = [(p,z,z_voxels,detector_rows, detector_columns, recon_volume,lock1,lock2) for z in xrange(z_voxels)]
        results =pool.map_async(fdkworker,args)
        results.get()
        #fdkworker(p,z,z_voxels,detector_rows, detector_columns, recon_volume,lock1,lock2)

    return recon_volume


def fdkcore(nr_projections, projections, combined_matrix, z_voxel_coords,
            transform_matrix, z_voxels, detector_rows, detector_columns,
            recon_volume, volume_weight, count_out):

    for p in xrange(nr_projections):
        if count_out: print 'Reconstructing projection: %s' % p
        
        # Numpy FDK operates on flat arrays

        flat_proj_data = projections[p].ravel()

        for z in xrange(z_voxels):

            # Put current z voxel into combined_matrix

            combined_matrix[2, :] = z_voxel_coords[z]

            # Find the mapping between volume voxels and detector pixels
            # for the current angle

            vol_det_map = dot(transform_matrix[p], combined_matrix)
            map_cols = rint(divide(vol_det_map[0, :], vol_det_map[2, :
                            ])).astype(int32)
            map_rows = rint(divide(vol_det_map[1, :], vol_det_map[2, :
                            ])).astype(int32)

            # Find the detector pixels that contribute to the current slice
            # xrays that hit outside the detector area are masked out

            mask = (map_cols >= 0) & (map_rows >= 0) & (map_cols
                    < detector_columns) & (map_rows < detector_rows)

            # The projection pixels that contribute to the current slice

            proj_indexs = map_cols * mask + map_rows * mask \
                * detector_columns

            # Add the weighted projection pixel values to their
            # corresponding voxels in the z slice

            recon_volume[z].flat += flat_proj_data[proj_indexs] \
                * volume_weight[p] * mask

    return recon_volume
