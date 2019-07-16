from multiprocessing.pool import Pool
from typing import List, Iterable

from gensim.models import KeyedVectors

from Clustering.abstract_cluster_builder import AbstractClusterBuilder
from Clustering.cluster import Cluster
from Clustering.similar_dimension_cluster_worker import SimilarDimensionClusterWorker


class SimilarDimensionClusterBuilder(AbstractClusterBuilder):

    def __init__(self, embedding: KeyedVectors, workers: int):
        super(SimilarDimensionClusterBuilder, self).__init__(embedding)
        self._number_of_workers = workers

    def run(self) -> List[Cluster]:
        dimensions: Iterable[int] = (dimension for dimension in range(self._embedding.vector_size))
        with Pool(processes=self._number_of_workers) as pool:
            clusters: List[Cluster] = pool.map(SimilarDimensionClusterWorker(self._embedding), dimensions)

        return [cluster for cluster in clusters if cluster is not None]
