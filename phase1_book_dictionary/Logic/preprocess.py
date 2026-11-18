import re
import string
import json
import os
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
import pandas as pd
from collections import defaultdict

from Logic.LSH import *

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
from ftfy import fix_text
class Preprocessor:
    def __init__(self, custom_stopwords_path='./stopwords.txt'):
        """
        Initialize the preprocessor, compile patterns, load components, etc.
        """
        base_dir = os.path.dirname(__file__)

        if custom_stopwords_path is None:
            custom_stopwords_path = os.path.join(
                base_dir,
                '..',
                'stopwords.txt'
            )

        self.pattern = re.compile(
            r'\S*http\S*|\S*www\S*|\S+\.ir\S*|\S+\.com\S*|\S+\.org\S*|\S*@\S*'
        )

        self.punctuation_table = str.maketrans(string.punctuation,' ' * len(string.punctuation))

        self.stemmer = PorterStemmer()

        self.stopwords = set(stopwords.words('english'))

        self.meta_data = {'author' , 'language'} #won't be stemmed or stopworded


    def preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing pipeline to a single text document.
        """
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = fix_text(text)
        text = re.sub(self.pattern, ' ', text)
        text = text.translate(self.punctuation_table)
        tokens = text.split()
        tokens = self.remove_stopwords(tokens)
            
        tokens = [self.normalize(token) for token in tokens]

        return " ".join(tokens)

    def remove_stopwords(self, tokens: str) -> list:
        """
        Remove stopwords from the text.
        """
        return [
            token for token in tokens
            if token not in self.stopwords
        ]
        
    
    def normalize(self, word: str) -> str:
        """
        Normalize the text by stemming, lemmatization, etc.

        Parameters
        ----------
        word : str
            The word to be normalized.

        Returns
        ----------
        list
            The normalized word.
        """
        return self.stemmer.stem(word)

    def preprocess_many(self, documents: list) -> list:
        """
        Apply preprocessing pipeline to a list of documents.
        """
        return [self.preprocess_text(doc) for doc in documents]
    


def preprocess_docs(docs: list):
    """
    Apply preprocessing to specific fields in a list of documents in-place.
    
    Args:
        docs (list): List of document dictionaries to preprocess
        
    Returns:
        None: Modifies the input list in-place
    
    Notes:
        Preprocesses the following fields: title, description, author
        Handles both string and list field types
    """
    preprocessor = Preprocessor()

    fields = ['title', 'description','author','genres', 'characters','language']

    for doc in docs:

        for field in fields:

            if field not in doc:
                continue

            value = doc[field]

            # string field
            if isinstance(value, str):
                doc[field] = preprocessor.preprocess_text(value)

            # list field
            elif isinstance(value, list):

                processed_list = []
                for item in value:
                    if isinstance(item, str):
                        processed = preprocessor.preprocess_text(item)
                        processed_list.extend(processed.split())
                doc[field] = list(dict.fromkeys(processed_list))


def csv_to_json(csv_file_path, json_file_path):
    """
    Convert a CSV file to JSON format with specific field mapping.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        json_file_path (str): Path where the output JSON file will be saved
        
    Returns:
        None: Writes output directly to JSON file
    
    Notes:
        Maps CSV fields to JSON structure including:
        - id (from bookId)
        - title, author, description
        - genres, characters, languages (split by commas)
        - publish_date, num_pages, avg_rating
    """
    books = []

    df = pd.read_csv(
        csv_file_path,
        encoding="utf-8",
        engine="python"
    ).fillna('')
    for _, row in df.iterrows():

        book = {

            "id": row.get("bookId", ""),

            "title": row.get("title", ""),

            "author": row.get("author", ""),

            "description": row.get("description", ""),

            "genres": str(row.get("genres", "")).split(","),

            "characters": str(row.get("characters", "")).split(","),

            "language": row.get("language", ""),

            "publish_date": row.get("publish_date", ""),

            "first_publish_date": row.get("first_publish_date", ""),

            "num_pages": row.get("num_pages", ""),

            "avg_rating": row.get("avg_rating", "")
        }

        books.append(book)

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(books, json_file, indent=4, ensure_ascii=False)



def normalize_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def safe_int(val):
    try:
        return int(val)
    except:
        return None


def safe_float(val):
    try:
        return float(val)
    except:
        return None
    
def metadata_similarity(doc1, doc2):

    score = 0
    total = 0

    # title
    t1 = normalize_text(doc1.get("title"))
    t2 = normalize_text(doc2.get("title"))
    if t1 and t2:
        total += 1
        if t1 == t2 or t1 in t2 or t2 in t1:
            score += 1

    # author
    a1 = normalize_text(doc1.get("author"))
    a2 = normalize_text(doc2.get("author"))
    if a1 and a2:
        total += 1
        if a1 == a2:
            score += 1

    # series
    s1 = normalize_text(doc1.get("series"))
    s2 = normalize_text(doc2.get("series"))
    if s1 and s2:
        total += 1
        if s1 == s2:
            score += 1

    # num_pages tolerance
    p1 = safe_int(doc1.get("num_pages"))
    p2 = safe_int(doc2.get("num_pages"))
    if p1 and p2:
        total += 1
        if abs(p1 - p2) <= 5:
            score += 1

    # avg rating tolerance
    r1 = safe_float(doc1.get("avg_rating"))
    r2 = safe_float(doc2.get("avg_rating"))
    if r1 and r2:
        total += 1
        if abs(r1 - r2) <= 0.15:
            score += 1

    # publication year
    y1 = str(doc1.get("first_publish_date", ""))[:4]
    y2 = str(doc2.get("first_publish_date", ""))[:4]
    if y1 and y2:
        total += 1
        if y1 == y2:
            score += 1

    if total == 0:
        return 0

    return score / total

def detect_metadata_duplicates(docs, threshold=0.75):

    duplicate_ids = set()

    # group by normalized title to avoid O(N²)
    title_groups = defaultdict(list)

    for i, doc in enumerate(docs):
        title = normalize_text(doc.get("title"))
        author = normalize_text(doc.get("author"))
        key = f"{title}|{author}"

        title_groups[key].append(i)

    for group in title_groups.values():

        if len(group) < 2:
            continue

        for i in range(len(group)):
            for j in range(i + 1, len(group)):

                id1 = group[i]
                id2 = group[j]

                if id2 in duplicate_ids:
                    continue

                sim = metadata_similarity(
                    docs[id1],
                    docs[id2]
                )

                if sim >= threshold:
                    duplicate_ids.add(id2)

    return duplicate_ids


def detect_isbn_duplicates(docs):

    seen = {}
    duplicates = set()

    for i, doc in enumerate(docs):

        isbn = doc.get("isbn13") or doc.get("isbn")

        if not isbn:
            continue

        if isbn in seen:
            duplicates.add(i)
        else:
            seen[isbn] = i

    return duplicates

def remove_duplicates(docs):

    print("Running duplicate detection...")

    # Step 1 — exact ISBN duplicates
    isbn_dups = detect_isbn_duplicates(docs)

    # Step 2 — metadata similarity
    metadata_dups = detect_metadata_duplicates(docs)

    all_duplicates = isbn_dups.union(metadata_dups)

    filtered_docs = [
        doc for i, doc in enumerate(docs)
        if i not in all_duplicates
    ]

    print("Removed duplicates:", len(all_duplicates))

    return filtered_docs


# def build_lsh_documents(docs):
#     """
#     Build LSH documents using only high-signal fields.
#     """

#     documents = []

#     for doc in docs:

#         parts = []

#         if isinstance(doc.get("title"), str):
#             parts.append(doc["title"])

#         if isinstance(doc.get("author"), str):
#             parts.append(doc["author"])

#         if isinstance(doc.get("description"), str):
#             parts.append(doc["description"])

#         documents.append(" ".join(parts))

#     return documents

# def remove_near_duplicates(docs):

#     documents = build_lsh_documents(docs)

#     lsh = MinHashLSH(documents, num_hashes=100)

#     buckets = lsh.perform_lsh()

#     duplicate_ids = set()

#     for bucket_docs in buckets.values():

#         unique_docs = list(set(bucket_docs))

#         if len(unique_docs) > 1:

#             for pair in itertools.combinations(unique_docs, 2):

#                 d1, d2 = pair

#                 s1 = lsh.shingle_document(documents[d1])
#                 s2 = lsh.shingle_document(documents[d2])

#                 score = lsh.jaccard_score(s1, s2)

#                 if score > 0.8:
#                     duplicate_ids.add(d2)

#     filtered_docs = [
#         doc for i, doc in enumerate(docs)
#         if i not in duplicate_ids
#     ]

#     print("Removed duplicates:", len(docs) - len(filtered_docs))

#     return filtered_docs





if __name__ == '__main__':

    base_dir = os.path.dirname(__file__)

    csv_path = os.path.join(base_dir, '..', 'top_3000_rated_books.csv')
    json_path = os.path.join(base_dir, 'crawled.json')

    csv_to_json(csv_path, json_path)

    
    json_file_path = json_path
    with open(json_file_path, "r", encoding="utf-8") as file:
        docs = json.load(file)

    preprocess_docs(docs)

    docs = remove_duplicates(docs)

    preprocessed_path = os.path.join(base_dir, 'preprocessed.json')

    with open(preprocessed_path, "w", encoding="utf-8") as file:
        json.dump(docs, file, indent=4, ensure_ascii=False)

