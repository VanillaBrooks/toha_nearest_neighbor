# toha_nearest_neighbor

Serial and parallel bindings to brute force and kd-tree methods for low dimensional (<16) 
nearest neighbor problems in python

[documentation](https://vanillabrooks.github.io/toha_nearest_neighbor/)

## Benchmarks

Some basic benchmarks have been carried out against `sklearn.neighbors` to show the relative performance
of this library. These preliminary results show better performance and better scaling in every function.
However, keep in mind that `sklearn` handles a generic n-dimensional space while this package 
has been simplified to work with 2D data. Moreover, `sklearn` also has an additional algorithm `ball_tree` that
scales to higher dimensions (N > 15) much better than kd-trees.

![](./static/benchmarks.png)
