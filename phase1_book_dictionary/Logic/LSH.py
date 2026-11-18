import numpy as np
import itertools
import random
import re

class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        document = re.sub(r"[^\w\s]", " ", document.lower())

        words = document.split()

        shingles = set()

        for i in range(len(words) - k + 1):
            shingles.add(" ".join(words[i:i + k]))

        return shingles
    
    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        doc_shingles = []
        all_shingles = set()

        for doc in self.documents:
            shingles = self.shingle_document(doc)
            doc_shingles.append(shingles)
            all_shingles.update(shingles)

        all_shingles = list(all_shingles)

        shingle_index = {s: i for i, s in enumerate(all_shingles)}

        num_shingles = len(all_shingles)
        num_docs = len(self.documents)

        matrix = np.zeros((num_shingles, num_docs), dtype=int)

        for doc_id, shingles in enumerate(doc_shingles):
            for shingle in shingles:
                matrix[shingle_index[shingle], doc_id] = 1

        return matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        char_matrix = self.build_characteristic_matrix()

        num_shingles, num_docs = char_matrix.shape

        signature = np.full((self.num_hashes, num_docs), np.inf)

        for i in range(self.num_hashes):

            perm = np.random.permutation(num_shingles)

            for row in range(num_shingles):

                permuted_row = perm[row]

                for doc in range(num_docs):
                    if char_matrix[permuted_row, doc] == 1:
                        signature[i, doc] = min(signature[i, doc], permuted_row)

        return signature

    def lsh_buckets(self, signature, bands=15, rows_per_band=4):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        buckets = {}

        num_docs = signature.shape[1]

        for band in range(bands):

            start = band * rows_per_band
            end = start + rows_per_band

            for doc_id in range(num_docs):

                band_slice = tuple(signature[start:end, doc_id])

                bucket_id = hash((band, band_slice))

                if bucket_id not in buckets:
                    buckets[bucket_id] = []
                buckets[bucket_id].append(doc_id)

        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        num_bands = 25

        signature = self.min_hash_signature()

        ans = self.lsh_buckets(
            signature,
            num_bands,
            self.num_hashes // num_bands
        )
        return ans

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = len(first_set & second_set)
        union = len(first_set | second_set)

        if union == 0:
            return 0

        return intersection / union

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

import os
import json
import itertools
import numpy as np
import random


def main():
    np.random.seed(42)
    random.seed(42)

    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, "LSHFakeData.json")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []

    for item in data:
        description = " ".join(item["descriptions"])
        documents.append(description)

    num_hashes = 100

    lsh = MinHashLSH(documents, num_hashes)

    buckets = lsh.perform_lsh()

    print("Number of buckets:", len(buckets))

    print("\nCandidate duplicate pairs:")

    detected_pairs = set()

    for bucket_docs in buckets.values():

        unique_docs = list(set(bucket_docs))

        if len(unique_docs) > 1:

            for pair in itertools.combinations(unique_docs, 2):

                pair = tuple(sorted(pair))

                if pair not in detected_pairs:
                    detected_pairs.add(pair)
                    print(pair)

    # ---------------------------------------------------
    # Ground truth duplicate pairs in fake dataset
    # ---------------------------------------------------

    true_pairs = {
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
        (12, 13),
        (14, 15),
        (16, 17),
        (18, 19)
    }

    # ---------------------------------------------------
    # Evaluation
    # ---------------------------------------------------

    correct_pairs = detected_pairs.intersection(true_pairs)

    precision = (
        len(correct_pairs) / len(detected_pairs)
        if len(detected_pairs) > 0 else 0
    )

    recall = (
        len(correct_pairs) / len(true_pairs)
        if len(true_pairs) > 0 else 0
    )

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print("\nEvaluation Metrics")
    print("-------------------")

    print("Detected pairs:", len(detected_pairs))
    print("True duplicate pairs:", len(true_pairs))
    print("Correct detections:", len(correct_pairs))

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")

    print()

    lsh.jaccard_similarity_test(buckets, documents)


if __name__ == '__main__':
    main()
