using Test
using HDBSCAN



@testset "HDBSCAN" begin

X = [randn(2,10000) fill(5,2).*randn(2,10000)]

result = hdbscan(X, min_cluster_size=60)

probabilities(result)
exemplars(result)       # Computed at first call, this takes long time
outlier_scores(result)
@test raw_data(result) == X

end
