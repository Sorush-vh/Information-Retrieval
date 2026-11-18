import time
import os
import json
import copy
from Logic.indexer.indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.CHARACTERS.value: self.index_characters(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.DESCRIPTIONS.value: self.index_descriptions(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {}
        for document in self.preprocessed_documents:
            current_index[document['id']] = document
        return current_index


    def index_characters(self):
        """
        Index the documents based on the characters.

        Returns
        ----------
        dict
            The index of the documents based on the characters. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        index = {}
        for document in self.preprocessed_documents:

            doc_id = document['id']

            characters = document.get('characters', [])

            for character in characters:

                terms = character if isinstance(character, list) else str(character).split()

                for term in terms:

                    if term not in index:
                        index[term] = {}

                    index[term][doc_id] = index[term].get(doc_id, 0) + 1
        return index



    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        index = {}
        for document in self.preprocessed_documents:

            doc_id = document['id']
            genres = document.get('genres', [])

            for genre in genres:

                terms = genre if isinstance(genre, list) else str(genre).split()

                for term in terms:

                    if term not in index:
                        index[term] = {}

                    index[term][doc_id] = index[term].get(doc_id, 0) + 1

        return index
  


    def index_descriptions(self):
        """
        Index the documents based on the descriptions.

        Returns
        ----------
        dict
            The index of the documents based on the descriptions. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        current_index = {}
        for document in self.preprocessed_documents:

            doc_id = document['id']

            description = document.get('description', [])

            if isinstance(description, str):
                terms = description.split()
            else:
                terms = description

            for term in terms:

                if term not in current_index:
                    current_index[term] = {}

                current_index[term][doc_id] = (
                    current_index[term].get(doc_id, 0) + 1
                )

        return current_index



    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, characters, genres, descriptions)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            return list(self.index[index_type].get(word, {}).keys())

        except:
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """
        doc_id = document['id']

        self.index[Indexes.DOCUMENTS.value][doc_id] = document

        for character in document.get('characters', []):

            terms = character if isinstance(character, list) else str(character).split()

            for term in terms:

                if term not in self.index[Indexes.CHARACTERS.value]:
                    self.index[Indexes.CHARACTERS.value][term] = {}

                self.index[Indexes.CHARACTERS.value][term][doc_id] = (
                    self.index[Indexes.CHARACTERS.value][term].get(doc_id, 0) + 1
                )

        for genre in document.get('genres', []):

            terms = genre if isinstance(genre, list) else str(genre).split()

            for term in terms:

                if term not in self.index[Indexes.GENRES.value]:
                    self.index[Indexes.GENRES.value][term] = {}

                self.index[Indexes.GENRES.value][term][doc_id] = (
                    self.index[Indexes.GENRES.value][term].get(doc_id, 0) + 1
                )

        description = document.get('description', [])

        if isinstance(description, str):
            description = description.split()

        for term in description:

            if term not in self.index[Indexes.DESCRIPTIONS.value]:
                self.index[Indexes.DESCRIPTIONS.value][term] = {}

            self.index[Indexes.DESCRIPTIONS.value][term][doc_id] = (
                self.index[Indexes.DESCRIPTIONS.value][term].get(doc_id, 0) + 1
            )



    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        document = self.index[Indexes.DOCUMENTS.value].get(document_id)

        if document is None:
            return

        del self.index[Indexes.DOCUMENTS.value][document_id]

        for character in document.get('characters', []):

            terms = character if isinstance(character, list) else str(character).split()

            for term in terms:

                postings = self.index[Indexes.CHARACTERS.value].get(term, {})

                postings.pop(document_id, None)

                if len(postings) == 0:
                    del self.index[Indexes.CHARACTERS.value][term]

        for genre in document.get('genres', []):

            terms = genre if isinstance(genre, list) else str(genre).split()

            for term in terms:

                postings = self.index[Indexes.GENRES.value].get(term, {})

                postings.pop(document_id, None)

                if len(postings) == 0:
                    del self.index[Indexes.GENRES.value][term]

        description = document.get('description', [])

        if isinstance(description, str):
            description = description.split()

        for term in description:

            postings = self.index[Indexes.DESCRIPTIONS.value].get(term, {})

            postings.pop(document_id, None)

            if len(postings) == 0:
                del self.index[Indexes.DESCRIPTIONS.value][term]


    def delete_dummy_keys(self, index_before_add, index, key):
        if len(index_before_add[index][key]) == 0:
            del index_before_add[index][key]

    def check_if_key_exists(self, index_before_add, index, key):
        if not index_before_add[index].__contains__(key):
            index_before_add[index].setdefault(key, {})


    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'characters': ['sandman', 'robin'],
            'genres': ['mystery', 'crime'],
            'description': 'good'
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return


        self.check_if_key_exists(index_before_add, Indexes.CHARACTERS.value, 'sandman')

        if (set(index_after_add[Indexes.CHARACTERS.value]['sandman']).difference(set(index_before_add[Indexes.CHARACTERS.value]['sandman']))
                != {dummy_document['id']}):
            print('Add is incorrect, sandman')
            return

        self.check_if_key_exists(index_before_add, Indexes.CHARACTERS.value, 'robin')

        if (set(index_after_add[Indexes.CHARACTERS.value]['robin']).difference(set(index_before_add[Indexes.CHARACTERS.value]['robin']))
                != {dummy_document['id']}):
            print('Add is incorrect, robin')
            return

        self.check_if_key_exists(index_before_add, Indexes.GENRES.value, 'mystery')

        if (set(index_after_add[Indexes.GENRES.value]['mystery']).difference(set(index_before_add[Indexes.GENRES.value]['mystery']))
                != {dummy_document['id']}):
            print('Add is incorrect, mystery')
            return

        self.check_if_key_exists(index_before_add, Indexes.GENRES.value, 'crime')

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        self.check_if_key_exists(index_before_add, Indexes.DESCRIPTIONS.value, 'good')

        if (set(index_after_add[Indexes.DESCRIPTIONS.value]['good']).difference(set(index_before_add[Indexes.DESCRIPTIONS.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        # Change the index_before_remove to its initial form if needed

        self.delete_dummy_keys(index_before_add, Indexes.CHARACTERS.value, 'sandman')
        self.delete_dummy_keys(index_before_add, Indexes.CHARACTERS.value, 'robin')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'mystery')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'crime')
        self.delete_dummy_keys(index_before_add, Indexes.DESCRIPTIONS.value, 'good')

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str = 'C:\\Users\\amir\\Desktop\\codes\\MIR-Project-SP2026\\answer\\indexes\\', index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, characters, genres, descriptions)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        file_path = os.path.join(path, index_name + '_index.json')

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(
                self.index[index_name],
                file,
                indent=4,
                ensure_ascii=False
            )


    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)


    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, characters, genres, descriptions)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """
        print('comparing indexes')
        
        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, characters, genres, descriptions)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the descriptions
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time < brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

def main():

    base_dir = os.path.dirname(__file__)

    preprocess_path = os.path.join(
        base_dir,
        '..',
        'preprocessed.json'
    )

    with open(preprocess_path, 'r', encoding='utf-8') as file:
        documents = json.load(file)

    index = Index(documents)

    index.check_add_remove_is_correct()

    index.check_if_indexing_is_good(
        Indexes.DESCRIPTIONS.value,
        'good'
    )

    indexes_path = os.path.join(
        base_dir,
        '..',
        '..',
        'indexes'
    )

    index.store_index(indexes_path, Indexes.DOCUMENTS.value)
    index.store_index(indexes_path, Indexes.CHARACTERS.value)
    index.store_index(indexes_path, Indexes.GENRES.value)
    index.store_index(indexes_path, Indexes.DESCRIPTIONS.value)

                




if __name__ == '__main__':
    main()
