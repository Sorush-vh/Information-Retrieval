import pickle
import os
#NOTE: COULD HAVE IMPLEMENTED A BETTER DATASTRUCTURE (I.E, TRIE) TO NOT LOOKUP
#ENTIRE DICTIONARY FOR SIMILIAR WORDS, AND INSTEAD CREATE TREE ON SHINGLES,
#AND ONLY TRAVERSE CANDIDATE MATCHES

class SpellCorrection:
    def __init__(self, all_documents=None, load_path=None, save_path=None):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str, optional
            The input documents used to build the vocabulary.
        load_path : str, optional
            Path to load precomputed data from.
        save_path : str, optional
            Path to save computed data to.
        """
        if load_path and os.path.exists(load_path):
            self.load(load_path)
        elif all_documents is not None:
            self.all_k_gram_words, self.word_counter = self.k_gramming_and_counting(all_documents)
            if save_path:
                self.save(save_path)
        else:
            self.all_k_gram_words = {}
            self.word_counter = {}

    def k_gram_word(self, word, k=2):
        """
        Convert a word into a set of k-grams.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each k-gram.

        Returns
        -------
        set
            A set of k-grams.
        """
        word = f"${word}$"
        kgrams = set()

        for i in range(len(word) - k + 1):
            kgrams.add(word[i:i+k])

        return kgrams

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of k-grams.
        second_set : set
            Second set of k-grams.

        Returns
        -------
        float
            Jaccard score.
        """
        intersection = len(first_set.intersection(second_set))
        union = len(first_set.union(second_set))

        if union == 0:
            return 0

        return intersection / union

    def k_gramming_and_counting(self, all_documents):
        """
        k-grams all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_k_gram_words : dict
            A dictionary from words to their k-grams sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_k_gram_words = {}
        word_counter = {}

        for doc in all_documents:

            if not isinstance(doc, str):
                continue

            words = doc.split()

            for word in words:

                word = word.lower()

                word_counter[word] = word_counter.get(word, 0) + 1

                if word not in all_k_gram_words:
                    all_k_gram_words[word] = self.k_gram_word(word)

        return all_k_gram_words, word_counter

    def save(self, path):
        """
        Save the k-grams data and word counter to a file.
        """
        data = {
            'all_k_gram_words': self.all_k_gram_words,
            'word_counter': self.word_counter
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        """
        Load the shingle data and word counter from a file.
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.all_k_gram_words = data['all_k_gram_words']
            self.word_counter = data['word_counter']

    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        word = word.lower()
        word_kgrams = self.k_gram_word(word)

        scores = []

        for vocab_word, vocab_kgrams in self.all_k_gram_words.items():

            jaccard = self.jaccard_score(word_kgrams, vocab_kgrams)

            if jaccard > 0:
                tf = self.word_counter.get(vocab_word, 0)
                final_score = jaccard * tf
                scores.append((vocab_word, final_score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scores[:5]]

    def spell_check(self, query,  min_jaccard=0.3):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        words = query.split()
        corrected = []

        for word in words:

            word_lower = word.lower()

            if word_lower in self.word_counter:
                corrected.append(word)
                continue

            candidates = []
            word_kgrams = self.k_gram_word(word_lower)

            for vocab_word, vocab_kgrams in self.all_k_gram_words.items():

                jaccard = self.jaccard_score(word_kgrams, vocab_kgrams)

                if jaccard >= min_jaccard:
                    tf = self.word_counter.get(vocab_word, 0)
                    final_score = jaccard * tf
                    candidates.append((vocab_word, final_score))

            candidates.sort(key=lambda x: x[1], reverse=True)

            if candidates and candidates[0][1] > 0.5:
                corrected.append(candidates[0][0])
            else:
                corrected.append(word)

        return " ".join(corrected)

