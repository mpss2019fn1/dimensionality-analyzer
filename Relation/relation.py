from typing import Dict


class Relation:

    def __init__(self, source: str, name: str, value: str):
        self.source: str = source
        self.name: str = name
        self.value: str = value

    @staticmethod
    def from_wikidata_record(record: Dict[str, str]) -> "Relation":
        return Relation(record["person"], record["wdLabel"], record["ps_Label"])

    @staticmethod
    def from_csv_record(record: Dict[str, str]) -> "Relation":
        return Relation(f"https://www.wikidata.org/wiki/{record['source']}", record["name"], record["value"])
