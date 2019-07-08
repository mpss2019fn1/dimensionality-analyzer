import statistics
from typing import List

from Clustering.abstract_cluster_builder import AbstractClusterBuilder
from Clustering.cluster import Cluster


class SimilarityDimensionClusterBuilder(AbstractClusterBuilder):

    def run(self) -> List[Cluster]:

        for dimension in range(self._embedding.vector_size):
            pass

    def extract_cluster(self, values: List[(str, float)]) -> List[str]:
        sorted_values: List[(str, float)] = sorted(values, key=lambda x : x[1])
        tolerance: float = statistics.mean(x[1] for x in sorted_values)
        largest_cluster: List[str] = []
        for start_index in range(len(sorted_values)):
            start_value: float = sorted_values[start_index][1]
            current_cluster: List[str] = list(sorted_values[start_index][0])
            for end_index in range(start_index + 1, len(sorted_values)):
                current_value: float = sorted_values[end_index][1]
                if current_value - start_value > tolerance:
                    break
                current_cluster.append(sorted_values[end_index][0])
            if len(current_cluster) > len(largest_cluster):
                largest_cluster = current_cluster
        return largest_cluster

