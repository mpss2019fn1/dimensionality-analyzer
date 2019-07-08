import csv
from pathlib import Path

from EntityLinking.entity_linking import EntityLinking
from FileParsing.abstract_file_parser import AbstractFileParser


class EntityLinkingFileParser(AbstractFileParser):
    COLUMN_INDEX_KNOWLEDGEBASE_ID = 1
    COLUMN_INDEX_EMBEDDING_LABEL = 0

    @staticmethod
    def create_from_file(configuration_file: Path) -> EntityLinking:
        with configuration_file.open("r") as csv_stream:
            csv_reader: csv.reader = csv.reader(csv_stream, delimiter=',')
            linking: EntityLinking = EntityLinking()

            next(csv_reader, None)  # skip header
            for row in csv_reader:
                if not row:
                    continue

                knowledgebase_id = row[EntityLinkingFileParser.COLUMN_INDEX_KNOWLEDGEBASE_ID]
                embedding_tag = row[EntityLinkingFileParser.COLUMN_INDEX_EMBEDDING_LABEL]
                linking.add(embedding_tag, knowledgebase_id)

            return linking
