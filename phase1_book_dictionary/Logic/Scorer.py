import math
from collections import Counter


class Scorer:
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The inverted index with structure {term: {document_id: tf}}.
        number_of_documents : int
            The number of documents in the collection.
        """
        self.index = index
        self.idf = {}
        self.N = max(int(number_of_documents), 1)
        self._collection_frequencies = None
        self._collection_length = None

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.
        """
        documents = set()

        for term in query:
            if term in self.index:
                documents.update(self.index[term].keys())

        return list(documents)

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.
        """
        if term in self.idf:
            return self.idf[term]

        df = len(self.index.get(term, {}))

        if df == 0:
            return 0

        value = math.log10(self.N / df)

        self.idf[term] = value

        return value

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.
        """
        return Counter(query)

    def compute_scores_with_vector_space_model(self, query, method):
        """
        Compute scores with vector space model.
        """
        document_method, query_method = method.split('.')

        scores = {}

        query_tfs = self.get_query_tfs(query)
        documents = self.get_list_of_documents(query)

        for document_id in documents:
            scores[document_id] = self.get_vector_space_model_score(
                query,
                query_tfs,
                document_id,
                document_method,
                query_method
            )

        return scores


    def get_vector_space_model_score(
        self, query, query_tfs, document_id, document_method, query_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.
        """
        document_weights = []
        query_weights = []

        unique_terms = set(query)

        for term in unique_terms:

            tf_d = self.index.get(term, {}).get(document_id, 0)
            tf_q = query_tfs.get(term, 0)

            doc_tf_weight = self._apply_tf(tf_d, document_method[0])
            query_tf_weight = self._apply_tf(tf_q, query_method[0])

            doc_idf = 1
            query_idf = 1

            if document_method[1] == 't':
                doc_idf = self.get_idf(term)

            if query_method[1] == 't':
                query_idf = self.get_idf(term)

            document_weights.append(doc_tf_weight * doc_idf)
            query_weights.append(query_tf_weight * query_idf)

        if document_method[2] == 'c':
            document_weights = self._cosine_normalize(document_weights)

        if query_method[2] == 'c':
            query_weights = self._cosine_normalize(query_weights)

        score = sum(
            d * q
            for d, q in zip(document_weights, query_weights)
        )

        return score


    def compute_socres_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        Compute scores with Okapi BM25.
        """
        scores = {}

        documents = self.get_list_of_documents(query)

        for document_id in documents:

            scores[document_id] = self.get_okapi_bm25_score(
                query,
                document_id,
                average_document_field_length,
                document_lengths
            )

        return scores

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.
        """
        k1 = 1.2
        b = 0.75

        score = 0

        dl = document_lengths.get(document_id, 0)

        for term in query:

            tf = self.index.get(term, {}).get(document_id, 0)

            if tf == 0:
                continue

            df = len(self.index.get(term, {}))

            idf = math.log10((self.N - df + 0.5) / (df + 0.5))

            numerator = tf * (k1 + 1)

            denominator = tf + k1 * (
                1 - b + b * (dl / average_document_field_length)
            )

            score += idf * (numerator / denominator)

        return score

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates scores for each document based on the unigram model.
        """
        self._prepare_collection_stats()

        scores = {}

        documents = self.get_list_of_documents(query)

        for document_id in documents:

            scores[document_id] = self.compute_score_with_unigram_model(
                query,
                document_id,
                smoothing_method,
                document_lengths,
                alpha,
                lamda
            )

        return scores

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the unigram score of a document for a query.
        """
        score = 0

        dl = document_lengths.get(document_id, 1)

        for term in query:

            tf = self.index.get(term, {}).get(document_id, 0)

            cf = self._collection_frequencies.get(term, 0)

            p_collection = cf / self._collection_length

            if smoothing_method == 'naive':

                probability = tf / dl if tf > 0 else 1e-10

            elif smoothing_method == 'bayes':

                probability = (
                    tf + alpha * p_collection
                ) / (dl + alpha)

            elif smoothing_method == 'mixture':

                probability = (
                    lamda * (tf / dl)
                    + (1 - lamda) * p_collection
                )

            else:
                raise ValueError("Invalid smoothing method")

            score += math.log(probability)

        return score

    def _apply_tf(self, tf, mode):
        """
        Apply term frequency (tf) weighting based on the specified mode.
        mode (str): Weighting scheme:
            - 'n'
            - 'l'

        """
        if mode == 'n':
            return tf

        elif mode == 'l':
            if tf > 0:
                return 1 + math.log10(tf)
            return 0

        else:
            raise ValueError("Invalid tf mode")

    def _cosine_normalize(self, weights):
        """
        Normalize a vector of term weights using cosine normalization.
        """
        norm = math.sqrt(sum(w * w for w in weights))

        if norm == 0:
            return weights

        return [w / norm for w in weights]

    def _prepare_collection_stats(self):
        """
        Compute and cache collection-wide statistics for the index.
        """
        if self._collection_frequencies is not None:
            return

        self._collection_frequencies = {}

        self._collection_length = 0

        for term, posting in self.index.items():

            cf = sum(posting.values())

            self._collection_frequencies[term] = cf

            self._collection_length += cf
