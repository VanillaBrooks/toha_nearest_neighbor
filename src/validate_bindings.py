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
                                                                    
# output: [0, 2]
out = toha_nearest_neighbor.brute_force(line_points, point_cloud)
print(out)
