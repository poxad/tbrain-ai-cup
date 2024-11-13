import json
import argparse
import pandas as pd
from rank_bm25 import BM25Okapi
import jieba
from stopwordsiso import stopwords
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
tokenizer = CkipWordSegmenter(model="bert-base")

# Convert the set of stop words to a list for compatibility with TfidfVectorizer
stop_words = list(stopwords(["zh"]))

def load_extracted_data(json_path):
    with open(json_path, 'r', encoding='utf8') as f:
        extracted_data = json.load(f)
    finance = {int(k): v for k, v in extracted_data["finance"].items()}
    insurance = {int(k): v for k, v in extracted_data["insurance"].items()}
    return finance, insurance

def remove_punctuation(text):
    text = re.sub('[!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~\n\s]', '', text)
    text = text.lower()  # Convert to lowercase
    return text

def remove_stop_words(text):
    words = jieba.cut(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ''.join(filtered_words)

def convert_gregorian_to_taiwan(text):
    gregorian_to_taiwan = {
        '2023年': '112年',
        '2022年': '111年',
        '2021年': '110年',
        # Add more mappings as needed
    }
    for greg_year, taiwan_year in gregorian_to_taiwan.items():
        text = text.replace(greg_year, taiwan_year)
    return text

def process_query(query):
    query_with_taiwan_years = convert_gregorian_to_taiwan(query)
    return query_with_taiwan_years

def tfidf_retrieve(query, source, corpus_dict):
    
    filtered_corpus = {key: corpus_dict[key] for key in source if key in corpus_dict}
    documents = list(filtered_corpus.values())
    doc_ids = list(filtered_corpus.keys())
    
    # Preprocess the documents (tokenize, remove punctuation, remove stop words)
    tokenized_documents = [' '.join(jieba.cut(remove_stop_words(remove_punctuation(doc)))) for doc in documents]
    # tokenized_documents = [' '.join(tokenizer([remove_stop_words(remove_punctuation(doc))])[0]) for doc in documents]
    
    # Initialize TF-IDF Vectorizer with custom stop words as a list
    vectorizer = TfidfVectorizer(
        stop_words=stop_words, 
        max_df=0.7,
        min_df=0.01, 
        ngram_range=(1, 3),
        sublinear_tf=True
    )
    
    # Fit and transform the filtered documents to create a TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(tokenized_documents)
    

    # Preprocess the query (remove punctuation, remove stop words, then tokenize)
    preprocessed_query = remove_stop_words(remove_punctuation(query))  # Remove punctuation and stopwords
    tokenized_query = ' '.join(jieba.cut(preprocessed_query))  # Tokenize the query
    # preprocessed_query = remove_stop_words(remove_punctuation(query))  # Remove punctuation and stopwords
    # tokenized_query = ' '.join(tokenizer([preprocessed_query])[0])  # Tokenize the query
    
    # Vectorize the query
    query_vector = vectorizer.transform([tokenized_query])
    
    # Compute cosine similarity between the query vector and all document vectors
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    best_match_idx = similarity_scores.argmax()
    best_match_id = doc_ids[best_match_idx]
    
    return best_match_id


if __name__ == "__main__":
    with open("dataset/questions_preliminary.json", 'rb') as f:
        qs_ref = json.load(f)

    corpus_dict_finance, corpus_dict_insurance = load_extracted_data("dataset/corpus/extracted_text.json")
    df_insurance = pd.DataFrame(list(corpus_dict_insurance.items()), columns=['pid', 'text'])
    df_finance = pd.DataFrame(list(corpus_dict_finance.items()), columns=['pid', 'text'])

    df_insurance = df_insurance.sort_values(by='pid').reset_index(drop=True)
    df_finance = df_finance.sort_values(by='pid').reset_index(drop=True)

    df_insurance['text'] = df_insurance['text'].apply(remove_punctuation).apply(remove_stop_words)
    df_finance['text'] = df_finance['text'].apply(remove_punctuation).apply(remove_stop_words)

    corpus_dict_insurance = dict(zip(df_insurance['pid'], df_insurance['text']))
    corpus_dict_finance = dict(zip(df_finance['pid'], df_finance['text']))

    with open('dataset/corpus/pid_map_content.json', 'rb') as f_s:
        key_to_source_dict = json.load(f_s)
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    answer_dict = {"answers": []}


    for q_dict in tqdm(qs_ref['questions'], ncols=100, desc="Processing Questions"):
        print(q_dict['query'])
        q_dict['query'] = process_query(q_dict['query'])
        if q_dict['category'] == 'insurance':
            retrieved = tfidf_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance)
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})
        else:
            print(f"Skipping question with unexpected category: {q_dict['category']}")
            continue

    with open("pred_retrieve_preliminary.json", 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
