[![Build Status](https://travis-ci.org/baggepinnen/HDBSCAN.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/HDBSCAN.jl)
[![codecov](https://codecov.io/gh/baggepinnen/HDBSCAN.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/HDBSCAN.jl)

# HDBSCAN
This package is a very simple wrapper around [hdbscan](https://github.com/scikit-learn-contrib/hdbscan) for python. It is not feature complete.

## Functions

```julia
result = hdbscan(data; kwargs...)

probabilities(result)
exemplars(result)
outlier_scores(result)
```
where `data` is `n_features × n_points` (the convention of Clustering.jl, opposite to the convention of the python library).

The `result::HdbscanResult` contains the `PyObject` clusterer which can be used to access everything that is not wrapped. 
