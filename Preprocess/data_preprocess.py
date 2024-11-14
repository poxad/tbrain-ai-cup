import os
import json
import pdfplumber
import numpy as np
from pdf2image import convert_from_path
import easyocr
from tqdm import tqdm


reader = easyocr.Reader(['ch_tra', 'en'])  # Initialize EasyOCR for Traditional Mandarin and English


def ocr_with_easyocr(pdf_path):
    """
    Perform OCR on a PDF file using EasyOCR.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.

    Notes:
        - Each page of the PDF is converted to an image for OCR processing.
        - OCR results are concatenated into a single string.
    """
    images = convert_from_path(pdf_path)  # Convert PDF to images
    pdf_text = ""
    for image in images:
        image_np = np.array(image)  # Convert to a NumPy array
        ocr_result = reader.readtext(image_np, detail=0)  # Get OCR text only
        pdf_text += " ".join(ocr_result) + "\n"
    return pdf_text


def read_pdf(pdf_loc):
    """
    Extract text from a PDF using `pdfplumber`. Fallback to OCR if extraction fails.

    Args:
        pdf_loc (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.

    Notes:
        - Structured text is extracted using `pdfplumber`.
        - If `pdfplumber` fails to extract meaningful text, OCR is used as a fallback.
        - The text is sorted to preserve logical reading order.
    """
    with pdfplumber.open(pdf_loc) as pdf:
        all_text = []  # List to store text from all pages

        # Iterate over all pages in the PDF
        for page in pdf.pages:
            # Extract words with their coordinates for the current page
            words = page.extract_words(use_text_flow=True)  # Preserve logical order

            # Sort words based on their position (top-to-bottom, left-to-right)
            sorted_words = sorted(words, key=lambda x: (x['top'], x['doctop'], x['x0']))

            # Combine the words into a single string for the current page
            page_text = ' '.join([word['text'] for word in sorted_words])
            all_text.append(page_text)  # Add page text to the list

        # Join all pages' text into a single string
        full_text = '\n'.join(all_text)

    if not full_text.strip():
        full_text = ocr_with_easyocr(pdf_loc)
    return full_text


def extract_text_from_folder(folder_path):
    """
    Process all PDF files in a folder and extract their text.

    Args:
        folder_path (str): Path to the folder containing PDF files.

    Returns:
        dict: A dictionary where keys are file identifiers (PID) and values are the extracted text.

    Notes:
        - PIDs are derived from the file names by removing the `.pdf` extension.
        - Non-PDF files in the folder are ignored.
        - The progress of the extraction is displayed using a progress bar.
    """
    corpus = {}
    for file in tqdm(os.listdir(folder_path), desc="Processing PDFs"):
        if file.endswith('.pdf'):
            file_path = os.path.join(folder_path, file)
            text = read_pdf(file_path)
            pid = int(file.replace('.pdf', ''))
            corpus[pid] = text
    return corpus


if __name__ == "__main__":
    """
    Main script execution.

    Steps:
        1. Extract text from the finance and insurance folders.
        2. Combine the extracted text into a single dictionary.
        3. Save the combined dictionary as a JSON file.

    Outputs:
        - A JSON file `extracted_text.json` containing text extracted from PDFs.

    Notes:
        - Ensure that the `finance` and `insurance` folders exist and contain valid PDF files.
    """
    # Paths for finance and insurance reference data
    finance_folder = "dataset/reference/finance"
    insurance_folder = "dataset/reference/insurance"

    # Extract text for each folder
    insurance_text = extract_text_from_folder(insurance_folder)
    finance_text = extract_text_from_folder(finance_folder)

    # Combine both dictionaries into a single JSON structure
    extracted_data = {
        "finance": finance_text,
        "insurance": insurance_text
    }

    # Save the extracted data to JSON
    with open("dataset/corpus/extracted_text.json", "w", encoding="utf8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)
