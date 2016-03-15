#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import dot, divide, int32, rint
import numpy as np
import pastset as ps
import multiprocessing as mp
workers = 4
cpus = 4
<<<<<<< HEAD

=======
>>>>>>> e03dd9f382acadac8f3774d6ba34bc265ba4e42b

def initPsets():
    pset = ps.PastSet()
    #projections, combined_matrix, z_voxel_coords,transform_matrix,volume_weight,z_voxels, detector_rows, detector_columns
    data = pset.enter(("data",np.array,np.array,np.array,np.array,np.array,int,int,int))
    #Make it a range?
<<<<<<< HEAD
    jobs = pset.enter(("jobs",int,int))
=======
    jobs = pset.enter(("jobs",int,int),fdkcorepset)
>>>>>>> e03dd9f382acadac8f3774d6ba34bc265ba4e42b
    #result set
    results = pset.enter(("results",np.array))

    return(pset,data,jobs,results)


def fdkcoreserver(nr_projections, projections, combined_matrix, z_voxel_coords,
            transform_matrix, z_voxels, detector_rows, detector_columns,
<<<<<<< HEAD
            recon_volume, volume_weight,pset,data,jobs,results):
    #pset,data,jobs,results = initPsets()

    #Initilize data. Only element in data
    data.move((projections,combined_matrix,z_voxel_coords,transform_matrix,
               volume_weight,z_voxels,detector_rows,detector_columns))
=======
            recon_volume, volume_weight):
    pset,data,jobs,results = initPsets()

    #Initilize data. Only element in data
    data.move((projections,combined_matrix,z_voxel_coords,transform_matrix,volume_weight,z_voxels,detector_rows,detector_columns))
>>>>>>> e03dd9f382acadac8f3774d6ba34bc265ba4e42b
    #splits = nr_projections

    #start workers before jobs, because they need time to load data anyway
    for i in xrange(workers):
<<<<<<< HEAD
        pset.spawn("fdkcore.py", "")

    #Dynamic split
    #splits = xrange(0,nr_projections,cpus)
    #for i in xrange(splits):
    #    jobs.move((i))

    #Static split on nodes
    workerSize = nr_projections/workers
    #A job specifying what data to process
    for i in xrange(workers):
        pstart = i * workerSize
        pend = (i+1) * workerSize if i != workers else nr_projections
        jobs.move((pstart,pend))
=======
        pset.spawn("fdkclient.py", "")


    splits = xrange(0,nr_projections,cpus)
    #A job specifying what data to process
    for i in splits:
        if  i + cpus >= nr_projections:
            jobs.move((i,nr_projections))
        else:
            jobs.move((i,i+cpus))
    #for i in xrange(splits):
    #    jobs.move((i))
>>>>>>> e03dd9f382acadac8f3774d6ba34bc265ba4e42b
    #Add poisen pills
    for i in xrange(workers):
        jobs.move((-1,0))

    for i in xrange(workers):
        recon_volume += results.observe()[0]
<<<<<<< HEAD
        #clean it up when possible. Should not be a problem
        results.axe(results.first()-1)

    data.axe(data.last()-1)
    #pset.halt()
=======
    pset.halt()
>>>>>>> e03dd9f382acadac8f3774d6ba34bc265ba4e42b



def fdkcoreclient():
    pset,data,jobs,results = initPsets()
    #Make variables global, so they work with X function below
    global projections,combined_matrix,z_voxel_coords,transform_matrix,volume_weight,z_voxels,detector_rows
    global detector_columns, recon_volume
    #Get and initialize data. Should not change
    projections,combined_matrix,z_voxel_coords,transform_matrix,volume_weight,z_voxels,detector_rows,detector_columns \
<<<<<<< HEAD
        = data.observe(data.last()-1)
    #'result' array
    recon_volume = np.zeros((z_voxels,z_voxels,z_voxels))

    pool = mp.Pool(cpus)
    lock = mp.Lock()

    #Works as poisen pill
    while True:# moreWork.observe():
        p = fdkcorepset(jobs.observe(),pool,lock)
        if p:
            break
=======
        = data.observe(0)
    #'result' array
    recon_volume = np.zeros((z_voxels,z_voxels,z_voxels))
    #Works as poisen pill
    while jobs.observe():# moreWork.observe():
        #Start next job.
        continue
>>>>>>> e03dd9f382acadac8f3774d6ba34bc265ba4e42b
    #When done, return result for merge
    results.move((recon_volume))
    pset.halt()

<<<<<<< HEAD
def fdkcorepsetInner(args):
    p,lock = args
    #Make local version
    comb_matrix = np.empty(combined_matrix.shape)
    comb_matrix[:] = combined_matrix[:]
=======

def fdkcorepset(args):
    #get arguments
    pstart = args[0]
    if pstart == -1:
        return False
    pend = args[1]
>>>>>>> e03dd9f382acadac8f3774d6ba34bc265ba4e42b

    for p in xrange(pstart,pend):
        # Numpy FDK operates on flat arrays

        flat_proj_data = projections[p].ravel()
        for z in xrange(z_voxels):

<<<<<<< HEAD
        comb_matrix[2, :] = z_voxel_coords[z]
=======
            # Put current z voxel into combined_matrix
>>>>>>> e03dd9f382acadac8f3774d6ba34bc265ba4e42b

            combined_matrix[2, :] = z_voxel_coords[z]

<<<<<<< HEAD
        vol_det_map = dot(transform_matrix[p], comb_matrix)
        map_cols = rint(divide(vol_det_map[0, :], vol_det_map[2, :
                        ])).astype(int32)
        map_rows = rint(divide(vol_det_map[1, :], vol_det_map[2, :
                        ])).astype(int32)
=======
            # Find the mapping between volume voxels and detector pixels
            # for the current angle
>>>>>>> e03dd9f382acadac8f3774d6ba34bc265ba4e42b

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

<<<<<<< HEAD
        # Add the weighted projection pixel values to their
        # corresponding voxels in the z slice
        with lock.accuire():
            recon_volume[z].flat += flat_proj_data[proj_indexs] \
                * volume_weight[p] * mask


def fdkcorepset(args,pool,lock):
    #get arguments
    pstart,pend = args
    #If is poisen pill, then stop
    if pstart == -1:
        return True





    for p in xrange(pstart,pend):
        #debugprint(p)
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

    return False
=======
            proj_indexs = map_cols * mask + map_rows * mask \
                * detector_columns

            # Add the weighted projection pixel values to their
            # corresponding voxels in the z slice

            recon_volume[z].flat += flat_proj_data[proj_indexs] \
                * volume_weight[p] * mask

    return True
>>>>>>> e03dd9f382acadac8f3774d6ba34bc265ba4e42b

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

def debugprint(p):
    with open("/home/ginger/Documents/cac/Ass3/test.txt", "w+") as f:
        f.write(str(p))

if __name__ == "__main__":
    fdkcoreclient()