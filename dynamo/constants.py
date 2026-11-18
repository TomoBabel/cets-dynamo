import re

TBL_EXT = ".tbl"
EM_EXT = ".em"
TBL_N_COLUMNS = 40
# Positions in a table (columns 24-26) are 1-based MATLAB voxel indices: dynamo_crop centres
# the box of a particle at position p on the voxels p-L/2..p+L/2-1. The CETS array frame is
# 0-based, so this offset is subtracted when reading and added when writing.
TBL_POSITION_OFFSET = 1
# Dynamo names the particle files after their tag (table column 1), e.g. particle_000001.em
# (zero-padded or not): the number at the end of the name.
PARTICLE_TAG_REGEX = re.compile(r"(\d+)\.em$")
