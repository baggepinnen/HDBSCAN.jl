[![Build Status](https://travis-ci.org/baggepinnen/HDBSCAN.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/HDBSCAN.jl)
[![codecov](https://codecov.io/gh/baggepinnen/HDBSCAN.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/HDBSCAN.jl)

# HDBSCAN
This package is a very simple wrapper around [hdbscan](https://github.com/scikit-learn-contrib/hdbscan) for python. It is not feature complete.

## Functions

```julia
using HDBSCAN, Clustering
result = hdbscan(X; min_cluster_size=5, min_samples=min_cluster_size, kwargs...)

probabilities(result)
exemplars(result)       # Computed at first call, this takes long time
outlier_scores(result)
```
where `X` is `n_features × n_points` (the convention of Clustering.jl, opposite to the convention of the python library).

The `result::HdbscanResult <: Clustering.ClusteringResult` contains the `PyObject` clusterer which can be used to access everything that is not wrapped.

The label assignments are stored in `result.assigments`. 0 values indicate noise (-1 in python version), positive values indicate a cluster assignment.

## Documentation
The original documentation is available here
https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
