.. toha_nearest_neighbor documentation master file, created by
   sphinx-quickstart on Tue Feb 21 16:06:54 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to toha_nearest_neighbor's documentation!
=================================================

.. _installation:

Installation
------------

Install with ``pip``:

.. code-block:: console

   $ pip3 install toha_nearest_neighbor


Api Reference
==================

.. autofunction:: toha_nearest_neighbor.brute_force_location(line_points: np.ndarray, point_cloud: np.ndarray, parallel=False) -> Tuple[np.ndarray, np.ndarray]

.. autofunction:: toha_nearest_neighbor.brute_force_index(line_points: np.ndarray, point_cloud: np.ndarray, parallel=False) -> Tuple[np.ndarray, np.ndarray]

.. autofunction:: toha_nearest_neighbor.kd_tree_location(line_points: np.ndarray, point_cloud: np.ndarray, parallel=False) -> Tuple[np.ndarray, np.ndarray]

.. autofunction:: toha_nearest_neighbor.kd_tree_index(line_points: np.ndarray, point_cloud: np.ndarray, parallel=False) -> Tuple[np.ndarray, np.ndarray]
