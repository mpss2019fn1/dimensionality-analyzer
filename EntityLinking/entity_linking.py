from typing import Dict


class EntityLinking:

    def __init__(self):
        self._entity_mappings: Dict[str, str] = {}

    def add(self, embedding_tag: str, knowledgebase_id: str) -> None:
        self._entity_mappings[knowledgebase_id] = embedding_tag

    def __getitem__(self, embedding_tag: str) -> str:
        return self._entity_mappings[embedding_tag]

    def __contains__(self, embedding_tag: str) -> bool:
        return embedding_tag in self._entity_mappings

    def __len__(self) -> int:
        return len(self._entity_mappings)
