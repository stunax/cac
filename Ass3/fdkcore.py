#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import dot, divide, int32, rint
import numpy as np
import pastset as ps
import multiprocessing as mp
workers = 4
cpus = 32

def initPsets():
    pset = ps.PastSet()
    #projections, combined_matrix, z_voxel_coords,transform_matrix,volume_weight,z_voxels, detector_rows, detector_columns
    data = pset.enter(("data",np.array,np.array,np.array,np.array,np.array,int,int,int))
    #Make it a range?
    jobs = pset.enter(("jobs",int,int))
    #result set
    results = pset.enter(("results",np.array))

    return(pset,data,jobs,results)


def fdkcoreserver(nr_projections, projections, combined_matrix, z_voxel_coords,
            transform_matrix, z_voxels, detector_rows, detector_columns,

            recon_volume, volume_weight,pset,data,jobs,results):
    #pset,data,jobs,results = initPsets()
    print workers,cpus
    #Initilize data. Only element in data
    data.move((projections,combined_matrix,z_voxel_coords,transform_matrix,volume_weight,z_voxels,detector_rows,detector_columns))
    #splits = nr_projections

    workerSize = nr_projections/workers

    #start new processes and a job for each
    for i in xrange(workers):
        pset.spawn("fdkcore.py", "")
        #Static split on nodes
        pstart = i * workerSize
        pend = (i+1) * workerSize if i != workers else nr_projections
        jobs.move((pstart,pend))

    #Dynamic split
    #splits = xrange(0,nr_projections,cpus)
    #for i in xrange(splits):
    #    jobs.move((i))

    #Add poisen pills
    for i in xrange(workers):
        jobs.move((-1,0))

    for i in xrange(workers):
        recon_volume += results.observe()[0]
        #clean it up when possible. Should not be a problem

    #clean up
    results.axe(results.first()-1)
    data.axe(data.last()-1)
    #pset.halt()

    return (cpus,workers)




def fdkcoreclient():
    pset,data,jobs,results = initPsets()
    #Make variables global, so they work with X function below
    global projections,combined_matrix,z_voxel_coords,transform_matrix,volume_weight,z_voxels,detector_rows
    global detector_columns, recon_volume, lock
    #Get and initialize data. Should not change
    projections,combined_matrix,z_voxel_coords,transform_matrix,volume_weight,z_voxels,detector_rows,detector_columns \
        = data.observe(data.last()-1)
    #'result' array
    recon_volume = np.zeros((z_voxels,z_voxels,z_voxels))

    pool = mp.Pool(cpus)

    #Works as poisen pill
    while True:# moreWork.observe():
        p = fdkcorepset(jobs.observe(),pool)
        if p:
            break

    #When done, return result for merge
    results.move((recon_volume))
    #pset.halt()


def fdkcorepsetInner(args):
    pstart,pend = args
    #Make local version
    comb_matrix = np.empty(combined_matrix.shape)
    comb_matrix[:] = combined_matrix[:]
    result  = np.zeros(recon_volume.shape)
    for p in xrange(pstart,pend):
        flat_proj_data = projections[p].ravel()
        for z in xrange(z_voxels):

            # Put current z voxel into combined_matrix
            comb_matrix[2, :] = z_voxel_coords[z]

            # Find the mapping between volume voxels and detector pixels
            # for the current angle
            vol_det_map = dot(transform_matrix[p], comb_matrix)
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

            result[z].flat += flat_proj_data[proj_indexs] * volume_weight[p] * mask
    return result


def fdkcorepset(args,pool):
    #get arguments from observe
    pstart,pend = args
    #If is poisen pill, then stop
    if pstart == -1:
        return True

    step = (pend-pstart) / cpus
    step = step if step > 0 else 1
    args = [(p,min(p+step,pend)) for p in range(pstart,pend,step)]

    calcres = pool.map_async(fdkcorepsetInner,args)
    res = calcres.get()
    recon_volume = sum(res)

    return False


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