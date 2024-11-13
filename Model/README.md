# Text Retrieval System Using TF-IDF, Jieba, and CKIP

This Python script is designed for text retrieval based on a query input, using a corpus extracted from two specific domains: **finance** and **insurance**. The text retrieval process utilizes the **TF-IDF** (Term Frequency-Inverse Document Frequency) method along with **jieba** for Chinese word segmentation and **CKIP Transformer** for better segmentation, tagging, and named entity recognition.

The script supports querying the domains of finance, insurance, and FAQ, processing textual data from PDFs and mapping the most relevant document using cosine similarity.

## Dependencies

- `jieba`: Chinese text segmentation
- `stopwordsiso`: To manage stop words for multiple languages, specifically Chinese (`zh`).
- `sklearn`: For TF-IDF vectorization and cosine similarity calculation
- `tqdm`: For progress bars during iterations
- `ckip-transformers`: For advanced word segmentation and NLP tasks
- `pandas`: For organizing extracted data into dataframes
- `json`: To load and save JSON data
- `re`: Regular expression operations for text cleaning
- `argparse`: To handle command-line arguments

## Purpose

The main objective of the script is to perform the following tasks:

1. **Text Preprocessing**: The script removes punctuation, numbers, stop words, and applies text normalization (e.g., converting Gregorian calendar years to the Taiwan calendar).
2. **Query Processing**: For each query, the system normalizes the query and processes it to make it compatible with the TF-IDF model.
3. **Text Retrieval**: For each question in the dataset, the script retrieves the most relevant document based on the cosine similarity between the processed query and the documents in the corpus.
4. **Category-Based Retrieval**: The script differentiates between documents from different categories, such as **finance**, **insurance**, and **FAQ**, and applies domain-specific preprocessing before querying.
5. **Output**: Results are saved as a JSON file with answers corresponding to the most relevant document for each question.

## Key Functions

### `load_extracted_data(json_path)`

- Loads the extracted text data from the provided JSON file (`extracted_text.json`).
- Returns two dictionaries containing the text from the **finance** and **insurance** categories.

### `remove_punctuation(text)`

- Removes punctuation marks from the input text.
- Converts the text to lowercase.

### `remove_punctuation_number(text)`

- Removes punctuation marks and numbers from the input text.
- Converts the text to lowercase.

### `remove_stop_words(text)`

- Removes stop words from the text using the **jieba** library for segmentation.

### `convert_gregorian_to_taiwan(text)`

- Converts Gregorian calendar years (e.g., 2023) to Taiwan years (e.g., 112 年).

### `process_query(query)`

- Processes the input query by converting Gregorian years to Taiwan years.

### `tfidf_retrieve(query, source, corpus_dict, category)`

- Retrieves the most relevant document from the corpus using **TF-IDF** and **cosine similarity**.
- Takes into account whether the query is related to **finance**, **insurance**, or **FAQ**, and preprocesses the text accordingly.

## Script Workflow

1. **Loading Data**: The script starts by loading the extracted data from the `extracted_text.json` file, which contains the text data for the finance and insurance domains.
2. **Processing Queries**: The script reads from a sample questions file (`questions_example.json`) and processes each question to prepare it for query-based retrieval.

3. **TF-IDF Model**: For each query, the script computes the TF-IDF vector for the query and compares it against the preprocessed corpus using **cosine similarity** to find the most relevant document.

4. **Results**: For each query, the best matching document ID is stored along with the question ID and returned as part of the output. The results are stored in `results/pred_retrieve_tfidf_ckip_and_jieba.json`.

## Usage

To run this script, you need to have the necessary dependencies installed. You can install them via `pip`:

```bash
pip install jieba stopwordsiso sklearn tqdm ckip-transformers pandas
```
