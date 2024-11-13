import os
import json
import pdfplumber
import numpy as np
from pdf2image import convert_from_path
import easyocr
from tqdm import tqdm


reader = easyocr.Reader(['ch_tra', 'en'])  # Initialize EasyOCR for Traditional Mandarin and English

def ocr_with_easyocr(pdf_path):
    images = convert_from_path(pdf_path)  # Convert PDF to images
    pdf_text = ""
    for image in images:
        image_np = np.array(image)  # Convert to a NumPy array
        ocr_result = reader.readtext(image_np, detail=0)  # Get OCR text only
        pdf_text += " ".join(ocr_result) + "\n"
    return pdf_text

def read_pdf(pdf_loc):
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
    corpus = {}
    for file in tqdm(os.listdir(folder_path)):
        if file.endswith('.pdf'):
            file_path = os.path.join(folder_path, file)
            text = read_pdf(file_path)
            pid = int(file.replace('.pdf', ''))
            corpus[pid] = text
    return corpus

if __name__ == "__main__":
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
