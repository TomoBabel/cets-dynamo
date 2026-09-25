from pathlib import Path

from dynamo.converters.subtomograms import DynamoSetOfSubtomograms
from dynamo.utils.utils import write_subtomograms_yaml

### DYNAMO TO CETS #################################################################
# Files
scratch_dir = Path("/home/jjimenez/CZII/cets_scratch_dir")
dynamo_run = Path(
    "/home/jjimenez/ScipionUserData/projects/czii_re5_extract_subtomos/"
    "Runs/001008_DynamoSubTomoMRA/extra"
)
tbl_file = dynamo_run / "initialTable.tbl"  # Dynamo table with the particle poses
particles_dir = dynamo_run / "data"  # directory with the extracted .em subvolumes
tomo_id = 0  # Dynamo numeric tomogram id (tbl column 20)
# Box voxel size in Å. Dynamo .em/.tbl files do not record a pixel size, so it must be
# supplied here (set it to the voxel size of the tomogram these particles were extracted at).
voxel_size = 5.4

# Subtomograms -> (PointSet3D, Average). The PointSet3D holds the picked coordinates
# (Region.annotations) and the Average holds the extracted ParticleMaps (Dataset.averages);
# each ParticleMap links back to a coordinate via
# source_region_id + source_annotation_id + coord_index.
dynamo_subtomo_set = DynamoSetOfSubtomograms()
result = dynamo_subtomo_set.dynamo_to_cets(
    tbl_file=tbl_file,
    dynamo_particles_directory=particles_dir,
    tomo_id=tomo_id,
    pixel_size=voxel_size,
)
if result is not None:
    point_set, average = result
    write_subtomograms_yaml(point_set, average, tomo_id, scratch_dir)
