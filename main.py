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
    logging.basicConfig(
        format='%(asctime)s : [Process-%(process)d] [Thread-%(threadName)s] %(levelname)s : %(message)s',
        level=logging.INFO)

    logging.info("loading embedding...")
    embedding: KeyedVectors = EmbeddingFileParser.create_from_file(args.embeddings)

    logging.info("loading linking...")
    linking: EntityLinking = EntityLinkingFileParser.create_from_file(args.linking)

    logging.info("building clusters...")
    cluster_builder: AbstractClusterBuilder = SimilarDimensionClusterBuilder(embedding=embedding, workers=args.workers)
    clusters: List[Cluster] = cluster_builder.run()

    with open(".clusters.txt", "w+") as clusters_file:
        for cluster in clusters:
            print(f"{cluster}", file=clusters_file)


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
    parser.add_argument(
        "--workers",
        type=int,
        help=f"Number of threads to use",
        required=False,
        default=8
    )
    parser.add_argument(
        "--use-cache",
        type=bool,
        help=f"Indicates, whether to use the relation cache",
        required=False,
        default=True
    )

    main(parser.parse_args())
