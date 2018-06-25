This folder contains demo code to play with the district-level adjacency matrices.

Below is a description of the adj_mats/ and spatial_indexes/ files.

 ---
 
Here are adjacency matrices for all of the states and the individual congressional districts at the level
of VTDs*.  

The data for Alaska and Hawaii is messy due to lots of disconnected land pieces, so it is not
recommended to make any conclusions based on analysis of these graphs.


District files are named XX_XXYY_dist.txt where XX is the state FIPS code (AL=01, AK=02, etc) and YY is
the district number.  For states with a single representative, YY=00.  These are written with comma separators
and carriage returns for newlines.

Load one of these with:

mat = np.loadtxt('01_0105_dist.txt',delimiter=',')

to load the Alabama 5th into an np.array called mat

The whole state adjacency matrices are saved as XX_full.npz where XX is the FIPS code.  .npz files are sparse
matrices and can be loaded into Python with the scipy.sparse library. the *_full.npz ones are of the whole
state and the *_blocks.npz are block-diagonal matrices with a block for each district.

mat = scipy.sparse.load_npz('06_full.npz')
to load the full California matrix into an np.array called mat

Each district has a spatial index to relate back to Census GIS files.  These are written as XX_XXYY_idx.txt with
the GEOID10, centroid.x, and centroid.y field on each line, and in the projection system used, this is (lon, lat).
The nth row in this file corresponds to the nth row/column in the XX_XXYY_dist file.
To get a spatial index for the whole state, you should concatenate each of the district-level index files in
consecutive order.

For now, there is no handling of disconnected/island parts, so be careful with graph analysis as some seemingly
connected regions may have more than one component.

The dual_graph folder contains visualizations.  Images for New York and Texas are forthcoming.


Kentucky, Oregon, Rhode Island, and Montana do not provide VTD data, so these graphs are constructed at the
level of county subdivisions.  The spatial index files for these states are aligned along the 'GEOID' column.
