## **AI CUP 2024 玉山人工智慧公開挑戰賽－RAG與LLM在金融問答的應用**

This repository implements a text retrieval system for the **2024 玉山人工智慧公開挑戰賽** (Yu Shan AI Challenge). The system is designed to extract text from PDF documents in two categories: **Finance** and **Insurance**, and retrieve the most relevant documents based on a query.

The solution involves two major steps:

1. **Data Preprocessing**: Extract text from PDF documents using `pdfplumber` and `easyocr`.
2. **Text Retrieval**: Use TF-IDF-based retrieval with Chinese text preprocessing for answering questions based on the extracted content.

---

### **Project Structure**

```
.
├── Preprocess
│   ├── data_preprocess.py            # Extract text from PDFs using pdfplumber and EasyOCR
│   └── README.md                    # Documentation for the preprocessing step
├── Model
│   ├── retrieval.py                  # Retrieve relevant documents using TF-IDF and cosine similarity
│   └── README.md                    # Documentation for the retrieval step
├── dataset                          # the needed dataset
│   ├── questions_preliminary.json
│   ├── corpus                        # contains extracted_text.json and pid_map_content.json
│   └── reference                     # contains the PDF files of finance and insurance (You can upload manually the PDF files)
├── main.py                           # Main entry point to run preprocessing and retrieval
├── requirements.txt                 # List of required Python packages
└── README.md                        # This file
```

---

### **Requirements**

This project requires Python 3.9.7 or later. It uses several Python libraries for text extraction, natural language processing, and machine learning. You can install the required dependencies using the provided `requirements.txt` file.

To set up the environment, first ensure that Python 3.9.7 or later is installed. Then, create a virtual environment and install the dependencies:

```bash
# Install the required dependencies
pip install -r requirements.txt
```

---

### **Installation and Setup**

1. **Clone the repository**:

   ```bash
   git clone https://github.com/poxad/tbrain-ai-cup.git
   cd tbrain-ai-cup
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your input PDFs**:

   - Place your PDF files in the `./reference/finance` and `./reference/insurance` folders. The system will process PDFs from these folders to extract text.

4. **Run the Project**:

---

### **Usage**

The project provides three primary functions, which can be executed via `main.py` using command-line arguments. For simplicity we have already preprocess the documents to the corpus called `extracted_text.json`, so you can just run the retrieval process. Here is the detail:

1. **Preprocess the Data**:

   - This step will extract text from the provided PDF files and save it as a JSON file (`extracted_text.json`).

   ```bash
   python main.py --preprocess
   ```

2. **Run the Retrieval Process**:

   - This step will retrieve the most relevant documents for a set of queries based on the preprocessed data. Results will be saved as `pred_retrieve_preliminary.json`.

   ```bash
   python main.py --retrieve
   ```

3. **Run the Full Workflow**:

   - This will first preprocess the data and then run the retrieval step.

   ```bash
   python main.py --preprocess --retrieve
   ```

---

### **File Descriptions**

- **`main.py`**: The main script to control the flow of the project. Use this to run data preprocessing and/or retrieval.
- **`data_preprocess.py`**: Extracts text from the PDF files in the `./reference/finance` and `./reference/insurance` folders and saves the extracted text in `extracted_text.json`.
- **`retrieval.py`**: Retrieves the most relevant documents for each query using TF-IDF and cosine similarity.
- **`requirements.txt`**: Lists the necessary Python libraries for this project.

---

### **Additional Information**

- The **Finance** and **Insurance** PDF files should be placed in the corresponding directories:
  - `./reference/finance/`
  - `./reference/insurance/`
- The **extracted_text.json** will store the extracted text data from these PDFs. It will be used during the retrieval process to find the most relevant documents for each query.

---

### **Example of Extracted Data Structure**

After running the data preprocessing step, the extracted data will be stored in `extracted_text.json` with the following structure:

```json
{
	"finance": {
		"0": "註 10： 本集團於民國 111 年第 1 季投資成...",
		"1": "國巨股份 瘩 其子公司 合忻裾攪羞表 民國..."
	},
	"insurance": {
		"1": "延期間內發生第十六條或第十七條本公司...",
		"10": "南山人壽新福愛小額終身壽險_MPL3 金」或喪葬費用保..."
	}
}
```

### **Example of Retrieval Output**

After running the retrieval step, the results will be saved in `pred_retrieve_preliminary.json` in the following format:

```json
{
  "answers": [
    {
      "qid": 5,
      "retrieve": 123
    },
    {
      "qid": 7,
      "retrieve": 456
    },
    ...
  ]
}
```

---

### **Dependencies in `requirements.txt`**

This project uses the following dependencies:

- **`pdfplumber==0.11.4`**: PDF text extraction library.
- **`tqdm`**: For showing progress bars.
- **`jieba==0.42.1`**: Chinese text segmentation.
- **`pdf2image==1.17.0`**: Converts PDFs to images for OCR.
- **`easyocr==1.7.2`**: OCR library for text recognition.
- **`stopwordsiso==0.6.1`**: Provides stopwords for multiple languages.
- **`scikit-learn==1.5.2`**: For machine learning and vectorization.
- **`ckip-transformers==0.3.4`**: For Chinese NLP tasks like word segmentation.
- **`os`, `json`, `numpy`, `pandas`**: For file management, JSON handling, and data processing.

---
