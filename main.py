import argparse
import logging
from pathlib import Path
from typing import Iterator

from gensim.models import KeyedVectors

from Clustering.abstract_cluster_builder import AbstractClusterBuilder
from Clustering.cluster import Cluster
from Clustering.similar_dimension_cluster_builder import SimilarDimensionClusterBuilder
from FileParsing.embedding_file_parser import EmbeddingFileParser


def main(args) -> None:
    embedding: KeyedVectors = EmbeddingFileParser.create_from_file(args.embeddings)
    cluster_builder: AbstractClusterBuilder = SimilarDimensionClusterBuilder(embedding=embedding)
    clusters: Iterator[Cluster] = cluster_builder.run()

    for cluster in clusters:
        print(cluster)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : [%(threadName)s] %(levelname)s : %(message)s', level=logging.INFO)

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        help=f"Path to the embeddings file (word2vec format)",
        required=True
    )
    main(parser.parse_args())
