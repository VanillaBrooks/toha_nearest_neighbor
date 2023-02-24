import toha_nearest_neighbor
import numpy as np
                                                                    
line_points = np.array(
    [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ]
)

point_cloud = np.array(
    [
        [0.1, -0.1], #closest to the 0-th index of line_points rows
        [2.2, 3.0], # closest to the 2-nd index of line_points rows
    ]
)

# indexes: [0 2],
# distances: [0.02, 1.04]
out = toha_nearest_neighbor.brute_force_index(line_points, point_cloud)
print(out)

# point locations:
# [[0., 0.],
#  [2., 2.]]
# distances:
# [0.02, 1.04]
out = toha_nearest_neighbor.kd_tree_location(line_points, point_cloud)
print(out)
