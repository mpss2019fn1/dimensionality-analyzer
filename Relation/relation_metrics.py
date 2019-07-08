from collections import Counter
from typing import Dict, Set, List, Tuple

from Relation.relation import Relation
from __resources__ import constant


class RelationMetrics:

    def __init__(self, number_of_entities: int):
        self._unique_relation_participants: Dict[str, Set[str]] = {}
        self._value_per_relation: Dict[str, Counter] = {}
        self._unique_relations_counter: Counter = Counter()
        self._number_of_entities: int = number_of_entities

    def add_relation(self, relation: Relation) -> None:
        if relation.name not in self._unique_relation_participants:
            self._unique_relation_participants[relation.name] = set()
            self._value_per_relation[relation.name] = Counter()
        if relation.source not in self._unique_relation_participants[relation.name]:
            self._unique_relations_counter[relation.name] += 1
            self._unique_relation_participants[relation.name].add(relation.source)
        self._value_per_relation[relation.name][relation.value] += 1

    def top_relations(self, max_relations: int, min_occurrence_factor: float = 0.3) -> List[Tuple[str, int]]:
        return list(filter(lambda x: x[1] > self._number_of_entities * min_occurrence_factor,
                           self._unique_relations_counter.most_common(max_relations)))

    def top_values(self, relation_name: str, max_values: int, min_occurrence_factor: float = 0.1) -> List[Tuple[str, int]]:
        return list(filter(lambda x: x[1] > self._number_of_entities * min_occurrence_factor,
                           self._value_per_relation[relation_name].most_common(max_values)))

    def __str__(self) -> str:
        representation = []
        for relation, relation_count in self.top_relations(constant.MAX_NUMBER_OF_RELATIONS_PER_CLUSTER):
            relation_percentage: float = round(relation_count / self._number_of_entities * 100, 2)
            representation.append(f"Relation: {relation} {relation_percentage}%")
            for value, value_count in self.top_values(relation, constant.MAX_NUMBER_OF_VALUES_PER_RELATION):
                value_percentage: float = round(value_count / self._number_of_entities * 100, 2)
                representation.append("\t↳ {:5.2f}% {}".format(value_percentage, value))

        return "\n".join(representation)
