TBL_EXT = ".tbl"
EM_EXT = ".em"
TBL_N_COLUMNS = 40
# Positions in a table (columns 24-26) are 1-based MATLAB voxel indices: dynamo_crop centres
# the box of a particle at position p on the voxels p-L/2..p+L/2-1. The CETS array frame is
# 0-based, so this offset is subtracted when reading and added when writing.
TBL_POSITION_OFFSET = 1
