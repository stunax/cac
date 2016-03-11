#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import float32, zeros
from fdkcore import fdkcore
from fdkload import fdkload
from time import time

def show(data):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.ion()
    for i in xrange(data.shape[0]):
        plt.clf()
        plt.matshow(data[i,:,:], fignum=1, cmap=cm.Greys_r)
        plt.draw()
        plt.pause(0.01)
        print "Showing: %d/%d" % (i+1, data.shape[0])

    plt.ioff()
    plt.close()
    
def main():
    
    # Initialize recon parameters

    nr_projections = 320
    detector_rows = 192
    detector_columns = 256

    voxels = x_voxels = y_voxels = z_voxels = 256
    
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

    print "Showing filtered projections"
    show(projections)
        
    # Reconstruct 3D Volume from recorded 2D images

    start = time()
    result = fdkcore(nr_projections, projections, combined_matrix,
                     z_voxel_coords, transform_matrix, z_voxels,
                     detector_rows, detector_columns, recon_volume,
                     volume_weight, count_out=True)
    stop = time()
    print 'System size', (x_voxels, y_voxels, z_voxels), 'Time taken', stop - start
    print "Showing reconstructed volume slices"
    show(result)
    
if __name__ == '__main__':
    main()
