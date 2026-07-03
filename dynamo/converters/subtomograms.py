import os
from typing import List, Tuple

import numpy as np

from cets_data_model.models.models import (
    PointSet3D,
    ParticleMap,
    AnnotationReference,
    Average,
    AnnotationType,
    Affine,
    CoordinateSystem,
    Axis,
    AxisType,
    Translation,
)
from cets_data_model.utils.image_utils import get_em_dims
from dynamo.constants import TBL_EXT
from dynamo.utils.utils import validate_file, get_particle_files, num_particles_in_tbl


# NOTE: SpaceAxis / AxisUnit enums were removed from the data model. Axis.name and
# Axis.axis_unit are now free-form strings; "ZXZ" (the Euler convention Dynamo uses)
# and "pixel" are preserved verbatim as the string equivalents of the old enums.
coordinates_system = [
    CoordinateSystem(
        name="Dynamo",
        axes=[Axis(name="ZXZ", axis_type=AxisType.space, axis_unit="pixel")],
    )
]


class DynamoSetOfSubtomograms:
    def dynamo_to_cets(
        self,
        tbl_file: os.PathLike,
        dynamo_particles_directory: os.PathLike,
        tomo_id: int,
    ) -> Tuple[PointSet3D, Average] | None:
        """Converts a set of subtomograms in Dynamo tbl format corresponding to the
        introduced tomogram identifier into CETS metadata.

        In the new data model the old ``Particle3D``/``Particle3DSet`` (which fused a picked
        coordinate and an extracted subvolume into one object) is split into two entities
        (Option A):

        * the picked coordinates -> a ``PointSet3D`` annotation (stored under
          ``Region.annotations``), linked to its tomogram via ``source_tomogram_id``;
        * each extracted subvolume -> a ``ParticleMap`` (stored under ``Average.particle_maps``),
          linked back to a single coordinate via ``source_annotation_reference_id`` +
          ``coord_index``.

        The bridge between the two is an ``AnnotationReference`` inside ``Average.annotations``
        that points at the ``PointSet3D`` (by region id + annotation id). ``coord_index`` is the
        0-based index into ``PointSet3D.origin3D``, so it must stay aligned with the order in
        which the coordinates are appended below.

        :param tbl_file: path to the tbl file containing the subtomograms data.
        :type tbl_file: os.PathLike.

        :param dynamo_particles_directory: path to the directory containing the
        subtomograms binary files (.em).
        :type dynamo_particles_directory: os.PathLike.

        :param tomo_id: Dynamo numeric identifier of the tomogram. It is used to indicate
        the tomogram from which the subtomograms will be converted, as in Dynamo the
        subtomograms from all the tomograms are stored together.
        :type tomo_id: int.
        """
        # Validate the tbl file
        validate_file(tbl_file, expected_ext=TBL_EXT)
        # Get the number of particles contained in the tbl file
        n_particles_in_tbl = num_particles_in_tbl(tbl_file)
        # Get the list of .em files contained in the particles directory provided
        dynamo_particle_files = get_particle_files(dynamo_particles_directory)
        n_particles_files = len(dynamo_particle_files)
        if n_particles_in_tbl != n_particles_files:
            raise Exception(
                f"The number of particles in the .tbl file provided [{n_particles_in_tbl}] "
                f"is different than the number of .em files [{n_particles_files}] contained "
                f"in the given particle files directory."
            )

        # TODO (open question #3): id-generation policy. tomo_id is reused as the
        # Region id and as the seed of the annotation/reference ids. These must be
        # unique within their respective scopes (Region within Dataset, Annotation
        # within Region.annotations, AnnotationReference within Average.annotations).
        annotation_id = f"dynamo_coords_{tomo_id}"
        reference_id = f"dynamo_ref_{tomo_id}"

        with open(tbl_file, "r") as dynamo_tbl:
            dynamo_particle_files = sorted(dynamo_particle_files)
            origin_3d: List[List[str]] = []
            particle_maps: List[ParticleMap] = []
            for ind, line in enumerate(dynamo_tbl):
                parts = line.split()
                vol_id = int(parts[19])
                if vol_id != tomo_id:
                    continue
                particle_fn = dynamo_particle_files[ind]
                size_x, size_y, size_z = get_em_dims(particle_fn)
                x = parts[23]
                y = parts[24]
                z = parts[25]
                origin_3d.append([x, y, z])
                particle_maps.append(
                    ParticleMap(
                        path=particle_fn,
                        width=size_x,
                        height=size_y,
                        depth=size_z,
                        source_annotation_reference_id=reference_id,
                        coord_index=len(particle_maps),
                        coordinate_transformations=[
                            self._get_particle_translation(parts),
                            self._get_particle_transform(parts),
                        ],
                    )
                )
            if not particle_maps:
                raise Exception(
                    f"No particle files were found matching the introduced Dynamo's "
                    f"tomogram numeric identifier [{tomo_id}]."
                )

            point_set = PointSet3D(
                id=annotation_id,
                name=f"Dynamo coordinates for {tomo_id}",
                annotation_type=AnnotationType.point_set_3D,
                source_tomogram_id=str(tomo_id),
                origin3D=origin_3d,
                coordinate_systems=coordinates_system,
            )
            average = Average(
                name=f"Dynamo subtomograms for {tomo_id}",
                annotations=[
                    AnnotationReference(
                        id=reference_id,
                        source_region_id=str(tomo_id),
                        source_annotation_id=annotation_id,
                    )
                ],
                particle_maps=particle_maps,
            )
            return point_set, average

    @staticmethod
    def _get_dynamo_euler_matrix(line_parts: List) -> np.ndarray:
        # .tbl rows are text, so the angle columns come in as strings; cast to float
        # before any numeric (numpy) operation.
        tdrot = float(line_parts[6])
        tilt = float(line_parts[7])
        narot = float(line_parts[8])
        # Convert the angles to radians
        tdrot = np.deg2rad(tdrot)
        tilt = np.deg2rad(tilt)
        narot = np.deg2rad(narot)
        # Rotations are clockwise
        tdrot *= -1
        tilt *= -1
        narot *= -1
        # Rotation around Z (tdrot)
        Rz1 = np.array(
            [
                [np.cos(tdrot), -np.sin(tdrot), 0],
                [np.sin(tdrot), np.cos(tdrot), 0],
                [0, 0, 1],
            ]
        )
        # Rotation around X (tilt)
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(tilt), -np.sin(tilt)],
                [0, np.sin(tilt), np.cos(tilt)],
            ]
        )
        # Rotation around Z (narot)
        Rz2 = np.array(
            [
                [np.cos(narot), -np.sin(narot), 0],
                [np.sin(narot), np.cos(narot), 0],
                [0, 0, 1],
            ]
        )
        # Compose ZXZ
        return Rz1 @ Rx @ Rz2

    def _get_particle_transform(self, line_parts: List) -> Affine:
        euler_matrix = self._get_dynamo_euler_matrix(line_parts)
        euler_matrix_list = euler_matrix.tolist()
        angular_matrix = [
            sublist[:3] for sublist in euler_matrix_list[:3]
        ]  # Take only the angular 3x3 sub-matrix
        return Affine(name="Subtomogram orientation", affine=angular_matrix)

    @staticmethod
    def _get_particle_translation(line_parts: List) -> Translation:
        shift_x = line_parts[3]
        shift_y = line_parts[4]
        shift_z = line_parts[5]
        return Translation(
            translation=[shift_x, shift_y, shift_z],
            name="Dynamo translation from a .tbl file. Shifts in pixels.",
        )
