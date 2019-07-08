import argparse
from pathlib import Path
from typing import List

from gensim.models import KeyedVectors

from Clustering.abstract_cluster_builder import AbstractClusterBuilder
from Clustering.cluster import Cluster
from Clustering.similarity_dimension_cluster_builder import SimilarityDimensionClusterBuilder
from FileParsing.embedding_file_parser import EmbeddingFileParser


def main(args) -> None:
    embedding: KeyedVectors = EmbeddingFileParser.create_from_file(args.embeddings)
    cluster_builder: AbstractClusterBuilder = SimilarityDimensionClusterBuilder(embedding=embedding)
    clusters: List[Cluster] = cluster_builder.run()


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        help=f"Path to the embeddings file (word2vec format)",
        required=True
    )
    main(parser.parse_args())
