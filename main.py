import argparse
import logging
from pathlib import Path
from typing import List

from gensim.models import KeyedVectors

from Clustering.abstract_cluster_builder import AbstractClusterBuilder
from Clustering.cluster import Cluster
from Clustering.similar_dimension_cluster_builder import SimilarDimensionClusterBuilder
from EntityLinking.entity_linking import EntityLinking
from FileParsing.embedding_file_parser import EmbeddingFileParser
from FileParsing.entity_linking_file_parser import EntityLinkingFileParser


def main(args) -> None:
    embedding: KeyedVectors = EmbeddingFileParser.create_from_file(args.embeddings)
    linking: EntityLinking = EntityLinkingFileParser.create_from_file(args.linking)

    cluster_builder: AbstractClusterBuilder = SimilarDimensionClusterBuilder(embedding=embedding)
    clusters: List[Cluster] = cluster_builder.run()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : [%(threadName)s] %(levelname)s : %(message)s', level=logging.INFO)

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        type=Path,
        help=f"Path to the embeddings file (word2vec format)",
        required=True
    )
    parser.add_argument(
        "--linking",
        type=Path,
        help=f"Path to the embedding linking file",
        required=True
    )

    main(parser.parse_args())
