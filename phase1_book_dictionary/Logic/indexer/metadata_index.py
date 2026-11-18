from Logic.indexer.index_reader import Index_reader
from Logic.indexer.indexes_enum import Indexes, Index_types
import json
import os

class Metadata_index:
    def __init__(self, path='index/'):
        """
        Initializes the Metadata_index.

        Parameters
        ----------
        path : str
            The path to the indexes.
        """
        
        self.documents = self.read_documents(path)

        self.metadata_index = self.create_metadata_index()

        self.store_metadata_index(path)


    def read_documents(self, path):
        """
        Reads the documents.
        
        """
        return Index_reader(path, index_name=Indexes.DOCUMENTS).index

    def create_metadata_index(self):    
        """
        Creates the metadata index.
        """
        metadata_index = {}
        metadata_index['averge_document_length'] = {
            'characters': self.get_average_document_field_length('characters'),
            'genres': self.get_average_document_field_length('genres'),
            'descriptions': self.get_average_document_field_length('description')
        }
        metadata_index['document_count'] = len(self.documents)

        return metadata_index
    
    def get_average_document_field_length(self,where):
        """
        Returns the sum of the field lengths of all documents in the index.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.
        """

        ans = 0
        for document in self.documents.values():

            field = document.get(where, [])

            if isinstance(field, str):

                ans += len(field.split())

            elif isinstance(field, list):

                for item in field:

                    if isinstance(item, list):
                        ans += len(item)
                    elif isinstance(item, str):
                        ans += len(item.split())

        return ans / max(len(self.documents), 1)



    def store_metadata_index(self, path):
        """
        Stores the metadata index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        """
        file_path = os.path.join(path, f"{Indexes.DOCUMENTS.value}_{Index_types.METADATA.value}_index.json")
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(self.metadata_index, file, indent=4)


    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    indexes_path = os.path.abspath(
        os.path.join(base_dir, "..", "..", "indexes")
    )

    Metadata_index(indexes_path)

    print("Metadata index stored successfully.")