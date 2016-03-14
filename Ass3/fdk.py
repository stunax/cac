#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import fromfile, float32, zeros
from fdkload import fdkload
from fdkcore import fdkcore
from time import time

def main():

    # Initialize recon parameters

    nr_projections = 320
    detector_rows = 192
    detector_columns = 256

    recon_voxel_sizes = [64, 128, 256]
        
    for voxels in recon_voxel_sizes:
        x_voxels = y_voxels = z_voxels = voxels

        # Load input data
        
        (projections, 
         combined_matrix, 
         z_voxel_coords, 
         transform_matrix,
         volume_weight) = fdkload(nr_projections,
                                  detector_rows,
                                  detector_columns,
                                  voxels)
                              
        # Initialize 3D volume data

        recon_volume = zeros((z_voxels, y_voxels, x_voxels), dtype=float32)

        # Reconstruct 3D Volume from recorded 2D images

        start = time()
        result = fdkcore(nr_projections, projections, combined_matrix,
                         z_voxel_coords, transform_matrix, z_voxels,
                         detector_rows, detector_columns, recon_volume,
                         volume_weight, count_out=False)
        #result = fdkcoreserver(nr_projections, projections, combined_matrix,
        #                 z_voxel_coords, transform_matrix, z_voxels,
        #                 detector_rows, detector_columns, recon_volume,
        #                 volume_weight)

        stop = time()
        print 'System size', (x_voxels, y_voxels, z_voxels), 'Time taken', stop - start

if __name__ == '__main__':
    main()
