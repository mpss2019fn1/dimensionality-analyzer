import csv
import os
from pathlib import Path
from typing import List, Dict

from EntityLinking.entity_linking import EntityLinking
from Relation.relation import Relation


class RelationFetcher:

    CACHE_FILE = Path(os.path.dirname(os.path.abspath(__file__)), "..", ".cached_relations.csv")

    def __init__(self, entity_linking: EntityLinking, cache_enabled: bool = True):
        self._linking: EntityLinking = entity_linking
        self._cache_enabled: bool = cache_enabled
        self._cached_relations: Dict[str, List[Relation]] = self._load_cached_relations()

    def _load_cached_relations(self) -> Dict[str, List[Relation]]:
        cached_relations: Dict[str, List[Relation]] = {}
        with self.CACHE_FILE.open("r") as input_stream:
            csv_reader: csv.reader = csv.reader(input_stream)
            next(csv_reader)  # skip header line

            for row in csv_reader:
                if not row:
                    continue  # ignore blank lines

            relation: Relation = Relation.from_csv_row(row)
            if relation.source not in cached_relations:
                cached_relations[relation.source] = []

            cached_relations[relation.source].append(relation)

        return cached_relations

    def _save_cached_relations(self) -> None:
        with self.CACHE_FILE.open("w+") as output_stream:
            print("source,name,target", file=output_stream)

            for relation_source in self._cached_relations:
                relations: List[Relation] = self._cached_relations[relation_source]

                for relation in relations:
                    print(f"{relation.source},{relation.name},{relation.value}", file=output_stream)

    def fetch(self, embedding_tags: List[str]) -> List[Relation]:
        entities: List[str] = embedding_tags
        relations: List[Relation] = []
        if self._cache_enabled:
            cached_entities = filter(lambda x: x not in self._cached_relations, entities)
            entities = list(set(entities) - set(cached_entities))
            cached_relations = (self._cached_relations[entity] for entity in cached_entities)
            relations.extend([relation for relations in cached_relations for relation in relations])  # list flatting
        # TODO: Add wikidataEndpoint

    def chunk_size(self) -> int:
        pass

    def __del__(self):
        self._save_cached_relations()
