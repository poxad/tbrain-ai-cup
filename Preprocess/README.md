# PDF Text Extraction from Folder

This Python script extracts text from PDF documents located in a specified folder. It uses two different methods for text extraction: `pdfplumber` for text-based PDFs and `EasyOCR` for image-based PDFs. The extracted text is then saved into a JSON file, which can be used for further processing or analysis.

## Purpose

The main goal of this script is to extract and consolidate text data from PDFs stored in two separate folders (finance and insurance), using the appropriate method based on whether the PDF contains text or images. The script is intended for use with financial and insurance reference data, but it can be adapted for other purposes as well.

## Functionality

### `ocr_with_easyocr(pdf_path)`
This function uses EasyOCR to perform Optical Character Recognition (OCR) on images extracted from a PDF. It converts each page of the PDF into an image and uses EasyOCR to extract text from these images.

- **Parameters:**
  - `pdf_path`: Path to the PDF file.
  
- **Returns:**
  - A string containing the text extracted from all images of the PDF.

### `read_pdf(pdf_loc)`
This function reads the text from a PDF using `pdfplumber`. It extracts the words on each page, preserving the logical flow of the text. If the text extraction is empty or insufficient, it falls back to using OCR with EasyOCR.

- **Parameters:**
  - `pdf_loc`: Path to the PDF file.
  
- **Returns:**
  - A string containing the text extracted from the PDF.

### `extract_text_from_folder(folder_path)`
This function iterates over all PDFs in a given folder, extracts the text from each file using the `read_pdf` function, and stores the results in a dictionary.

- **Parameters:**
  - `folder_path`: Path to the folder containing PDF files.
  
- **Returns:**
  - A dictionary where the keys are the PDF IDs (extracted from the filenames) and the values are the corresponding extracted text.

### Main Script
The main script extracts text from two folders: one for finance-related PDFs and one for insurance-related PDFs. It then combines the extracted text from both folders into a single JSON structure and saves it to a file named `extracted_text.json`.

- **Folders processed:**
  - `./reference/finance`: Contains finance-related PDFs.
  - `./reference/insurance`: Contains insurance-related PDFs.

- **Output:**
  - `extracted_text.json`: A JSON file containing the extracted text from both the finance and insurance PDFs.

## Requirements

- `easyocr`: For Optical Character Recognition on image-based PDFs.
- `pdfplumber`: For extracting text from text-based PDFs.
- `pdf2image`: For converting PDF pages to images.
- `numpy`: For handling image data.
- `tqdm`: For progress bar during folder processing.

## Installation

To install the required dependencies, run:

```bash
pip install easyocr pdfplumber pdf2image numpy tqdm
