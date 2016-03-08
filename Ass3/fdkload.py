#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import fromfile, float32

def fdkload(nr_projections, 
            detector_rows, 
            detector_columns,
            voxels):

    x_voxels = y_voxels = z_voxels = voxels
    
    # Load 2D projection data

    projections = fromfile('input/projections.bin', dtype=float32)
    projections.shape = (nr_projections, detector_rows,
                         detector_columns)

    # Load combined X,Y voxel coordinates

    combined_matrix = fromfile('input/%s/combined.bin' % voxels, dtype=float32)
    combined_matrix.shape = (4, y_voxels * x_voxels)

    # Load Z voxel coordinates

    z_voxel_coords = fromfile('input/%s/z_voxel_coords.bin' % voxels, dtype=float32)

    # Load transform matrix used to align the 3D volume position
    # towards the recorded 2D projection

    transform_matrix = fromfile('input/transform.bin', dtype=float32)
    transform_matrix.shape = (nr_projections, 3, 4)

    # Load volume weight used to compensate for conebeam ray density

    volume_weight = fromfile('input/%s/volumeweight.bin' % voxels, dtype=float32)
    volume_weight.shape = (nr_projections, y_voxels * x_voxels)


    return (projections, combined_matrix, z_voxel_coords, transform_matrix, volume_weight)
