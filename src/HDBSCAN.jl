module HDBSCAN
using PyCall, Clustering
export hdbscan, probabilities, exemplars, outlier_scores

hdbs = pyimport("hdbscan")

struct HdbscanResult <: ClusteringResult
    clusterer
    assignments
end

function hdbscan(X; kwargs...)
    clusterer = hdbs.HDBSCAN(; kwargs...)
    cluster_labels = clusterer.fit_predict(X')
    HdbscanResult(clusterer, cluster_labels .+ 1)
end

probabilities(cr::HdbscanResult) = cr.clusterer.probabilities_
exemplars(cr::HdbscanResult) = cr.clusterer.exemplars_
outlier_scores(cr::HdbscanResult) = cr.clusterer.outlier_scores_

end
