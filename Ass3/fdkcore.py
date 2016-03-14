#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import dot, divide, int32, rint
import numpy as np
import pastset as ps
workers = 4

def initPsets():
    pset = ps.PastSet()
    #projections, combined_matrix, z_voxel_coords,transform_matrix,volume_weight,z_voxels, detector_rows, detector_columns
    data = pset.enter(("data",np.array,np.array,np.array,np.array,np.array,int,int,int))
    moreWork = pset.enter(("moreWork",bool))
    #Make it a range?
    jobs = pset.enter(("jobs",int),fdkcorepset)
    #result set
    results = pset.enter(("results",np.array))

    return(pset,data,moreWork,jobs,results)


def fdkcoreserver(nr_projections, projections, combined_matrix, z_voxel_coords,
            transform_matrix, z_voxels, detector_rows, detector_columns,
            recon_volume, volume_weight):
    pset,data,moreWork,jobs,results = initPsets()

    #Initilize data. Only element in data
    data.move((projections,combined_matrix,z_voxel_coords,transform_matrix,volume_weight,z_voxels,detector_rows,detector_columns))
    splits = nr_projections

    #Is there more work?
    for i in xrange(splits):
        moreWork.move((True))
    for i in xrange(workers):
        moreWork.move((False))
        pset.spawn("fdkclient.py", "")

    #A job specifying
    for i in xrange(splits):
        jobs.move((i))


    for i in xrange(workers):
        recon_volume += results.observe()



def fdkcoreclient():
    pset,data,moreWork,jobs,results = initPsets()
    #Make variables global, so they work with X function below
    global projections,combined_matrix,z_voxel_coords,transform_matrix,volume_weight,z_voxels,detector_rows
    global detector_columns, recon_volume
    #Get and initialize data. Should not change
    projections,combined_matrix,z_voxel_coords,transform_matrix,volume_weight,z_voxels,detector_rows,detector_columns = data.first()
    #'result' array
    recon_volume = np.zeros(z_voxels,z_voxels,z_voxels)
    #Check if everything has been initiated
    while moreWork.observe():
        #Start next job.
        jobs.observe()
    #When done, return result for merge
    results.move((recon_volume))


def fdkcorepset(p):
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
