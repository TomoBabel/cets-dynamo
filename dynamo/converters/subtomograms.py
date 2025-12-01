import os
from typing import List

import numpy as np

from cets_data_model.models.models import (
    Particle3D,
    Affine,
    CoordinateSystem,
    Axis,
    SpaceAxis,
    AxisType,
    AxisUnit,
    Translation,
    Particle3DSet,
)
from cets_data_model.utils.image_utils import get_em_file_info
from dynamo.constants import TBL_EXT
from dynamo.utils.utils import validate_file, get_particle_files, num_particles_in_tbl


coordinates_system = [
    CoordinateSystem(
        name="Dynamo",
        axes=[
            Axis(name=SpaceAxis.ZXZ, axis_type=AxisType.space, axis_unit=AxisUnit.pixel)
        ],
    )
]


class DynamoSetOfSubtomograms:
    def dynamo_to_cets(
        self,
        tbl_file: os.PathLike,
        dynamo_particles_directory: os.PathLike,
        tomo_id: int,
    ) -> Particle3DSet:
        """Converts a set of subtomograms in Dynamo tbl format corresponding to the
        introduced tomogram identifier into CETS metadata.

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
        if n_particles_in_tbl != dynamo_particle_files:
            raise Exception(
                f"The number of particles in the .tbl file provided [{n_particles_in_tbl}] "
                f"is different than the number of .em files [{n_particles_files}] contained "
                f"in the given particle files directory."
            )

        with open(tbl_file, "r") as dynamo_tbl:
            dynamo_particle_files = sorted(dynamo_particle_files)
            particle_list = []
            for ind, line in enumerate(dynamo_tbl):
                parts = line.split()
                vol_id = int(parts[19])
                if vol_id != tomo_id:
                    continue
                particle_fn = dynamo_particle_files[ind]
                img_file_info = get_em_file_info(particle_fn)
                x = parts[23]
                y = parts[24]
                z = parts[25]
                particle_list.append(
                    Particle3D(
                        path=particle_fn,
                        position=[x, y, z],
                        width=img_file_info.size_x,
                        height=img_file_info.size_y,
                        depth=img_file_info.size_z,
                        coordinate_transformations=[
                            self._get_particle_translation(parts),
                            self._get_particle_transform(parts),
                        ],
                    )
                )
            if not particle_list:
                raise Exception(
                    f"No particle files were found matching the introduced Dynamo's "
                    f"tomogram numeric identifier [{tomo_id}]."
                )
            particles = Particle3DSet(
                particles=particle_list,
                coordinate_systems=coordinates_system,
            )
            return particles

    @staticmethod
    def _get_dynamo_euler_matrix(line_parts: List) -> np.ndarray:
        tdrot = line_parts[6]
        tilt = line_parts[7]
        narot = line_parts[8]
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
