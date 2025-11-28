import os
from dynamo.constants import TBL_EXT
from dynamo.utils.utils import validate_file, get_particle_files, num_particles_in_tbl


class DynamoSetOfSubtomograms:
    @staticmethod
    def dynamo_to_cets(
        tbl_file: os.PathLike, dynamo_particles_directory: os.PathLike, tomo_id: int
    ):
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
            # particle_list = []
            for line in dynamo_tbl:
                parts = line.split()
                vol_id = int(parts[19])
                if vol_id != tomo_id:
                    continue
                # shift_x = parts[3]
                # shift_y = parts[4]
                # shift_z = parts[5]
                # tdrot = parts[6]
                # tilt = parts[7]
                # narot = parts[8]
                # angle_min = parts[13]
                # angle_max = parts[14]
                # class_id = parts[21]
                #
                # x = parts[23]
                # y = parts[24]
                # z = parts[25]

                # particle3d = Particle3D()

            # if not particle_list:
            #     raise Exception(
            #         f"No particle files were found matching the introduced Dynamo's "
            #         f"tomogram numeric identifier [{tomo_id}]."
            #     )


# import numpy as np
#
#
# def dynamo_orientation_matrix(tdrot, tilt, narot, degrees=True):
#     """
#     Calcula la matriz de orientación de Dynamo (Euler ZXZ).
#
#     Parámetros:
#         tdrot (float): Primer ángulo (rotación sobre Z)
#         tilt (float): Segundo ángulo (rotación sobre X)
#         narot (float): Tercer ángulo (rotación sobre Z)
#         degrees (bool): True si los ángulos están en grados (por defecto Dynamo usa grados)
#
#     Retorna:
#         numpy.ndarray: Matriz de rotación 3x3
#     """
#     # Conversión a radianes si es necesario
#     if degrees:
#         tdrot = np.deg2rad(tdrot)
#         tilt = np.deg2rad(tilt)
#         narot = np.deg2rad(narot)
#
#     # Dynamo aplica rotaciones en sentido horario → signo negativo
#     tdrot *= -1
#     tilt *= -1
#     narot *= -1
#
#     # Rotación sobre Z (tdrot)
#     Rz1 = np.array(
#         [
#             [np.cos(tdrot), -np.sin(tdrot), 0],
#             [np.sin(tdrot), np.cos(tdrot), 0],
#             [0, 0, 1],
#         ]
#     )
#
#     # Rotación sobre X (tilt)
#     Rx = np.array(
#         [[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]]
#     )
#
#     # Rotación sobre Z (narot)
#     Rz2 = np.array(
#         [
#             [np.cos(narot), -np.sin(narot), 0],
#             [np.sin(narot), np.cos(narot), 0],
#             [0, 0, 1],
#         ]
#     )
#
#     # Composición ZXZ
#     return Rz1 @ Rx @ Rz2
#
#
# alpha = -132.33
# beta = 78.844
# gamma = -72.009
# M = dynamo_orientation_matrix(alpha, beta, gamma)
# print(M)
