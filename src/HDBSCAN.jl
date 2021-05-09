module HDBSCAN
using PyCall, Clustering
export hdbscan, probabilities, exemplars, outlier_scores, leaf_size, raw_data

const hdbs = PyNULL()

function __init__()
    copy!(hdbs, pyimport("hdbscan"))
end

"""

hdbscan(min_cluster_size=5, min_samples=min_cluster_size, metric="euclidean", alpha=1.0, p=nothing, algorithm="best", leaf_size=40, memory=Memory(location=nothing), approx_min_span_tree=true, gen_min_span_tree=false, core_dist_n_jobs=4, cluster_selection_method="eom", allow_single_cluster=false, prediction_data=false, match_reference_implementation=false)
"""
function hdbscan(X; min_cluster_size=5, kwargs...)
    @show hdbs
    clusterer = hdbs.HDBSCAN(;min_cluster_size=min_cluster_size, kwargs...)
    cluster_labels = clusterer.fit_predict(X')
    HdbscanResult(clusterer, cluster_labels .+ 1)
end

struct HdbscanResult <: ClusteringResult
    clusterer
    assignments
end

# @show hdbs.HDBSCAN(randn(2,100))


probabilities(cr::HdbscanResult) = cr.clusterer.probabilities_
exemplars(cr::HdbscanResult) = cr.clusterer.exemplars_
outlier_scores(cr::HdbscanResult) = cr.clusterer.outlier_scores_
leaf_size(cr::HdbscanResult) = cr.clusterer.leaf_size_
raw_data(cr::HdbscanResult) = cr.clusterer._raw_data |> transpose

end
