# scripts/pdf_extractor.py

import os
import json
import sys
import fitz  # PyMuPDF
from typing import Dict, Any, List, Tuple
import pdfplumber
import io
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
# pdfplumber re‑raises most parsing issues as pdfminer.six exceptions
from pdfminer.pdfparser import PDFSyntaxError
import logging

# Configure logging
logger = logging.getLogger(__name__)

def build_sections_from_toc(toc: List[List], total_pages: int) -> List[Dict[str, Any]]:
    sections = []
    for i, entry in enumerate(toc):
        level, title, start_page = entry
        if i < len(toc) - 1:
            next_start = toc[i + 1][2]
            end_page = next_start - 1
        else:
            end_page = total_pages
        sections.append({
            "title": title,
            "start_page": start_page,
            "end_page": end_page,
            "method": "TOC"
        })
    return sections

def build_sections_from_layout(pdf_path: str,
                               font_size_threshold: float = 14.0) -> List[Dict[str, Any]]:
    """
    Attempt to infer section breaks by scanning for heading‑sized fonts.
    If pdfplumber cannot open the file (corrupted / non‑PDF wrapper),
    gracefully fall back by returning an empty list so the caller can
    revert to page‑based segmentation.
    """
    candidate_headings = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                words = page.extract_words(extra_attrs=["size", "fontname"])
                for word in words:
                    size = word.get("size", 0.0)
                    text = word.get("text", "")
                    if (
                        size >= font_size_threshold and
                        any(kw.lower() in text.lower()
                            for kw in ("chapter", "section", "part", "Chapter", "Section", "Part"))
                    ):
                        candidate_headings.append({
                            "page": page.page_number,
                            "text": text,
                            "font_size": size
                        })
            candidate_headings.sort(key=lambda x: x["page"])
    except PDFSyntaxError as e:
        # Most commonly: "No /Root object! - Is this really a PDF?"
        logger.warning(f"PDF syntax error in '{pdf_path}': {e}")
        return []
    except FileNotFoundError as e:
        logger.error(f"PDF file not found: '{pdf_path}'")
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing '{pdf_path}': {e}")
        return []

    sections = []
    if candidate_headings:
        for i, h in enumerate(candidate_headings):
            start_page = h["page"]
            end_page = (candidate_headings[i + 1]["page"] - 1
                        if i < len(candidate_headings) - 1 else total_pages)
            sections.append({
                "title": h["text"],
                "start_page": start_page,
                "end_page": end_page,
                "method": "Layout"
            })
    return sections

# ────────────────────────────────────────────────
# OCR and multi‑column handling helpers
# ────────────────────────────────────────────────
def ocr_page_words(page, dpi: int = 350, lang: str = "eng") -> pd.DataFrame:
    """Render a page to high‑DPI PNG and return a DataFrame of word boxes."""
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    # --- Pre-process for OCR: grayscale + Otsu binarization ---
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = Image.fromarray(thresh)
    df = pytesseract.image_to_data(
        img,
        lang=lang,
        config="--oem 1 --psm 3 -c preserve_interword_spaces=1",
        output_type=pytesseract.Output.DATAFRAME
    )
    df = df[(df.conf != -1) & df.text.notnull()].copy()
    df.rename(columns={"left": "x0", "top": "y0"}, inplace=True)
    df["x1"] = df.x0 + df.width
    df["y1"] = df.y0 + df.height
    return df[["x0", "y0", "x1", "y1", "text"]]

def is_multicol(df: pd.DataFrame, page_width: float, gap_ratio_thr: float = 0.15) -> bool:
    """Return True if the page likely has multiple text columns."""
    if len(df) < 30:
        return False
    centers = ((df.x0 + df.x1) / 2).to_numpy()
    centers.sort()
    gaps = np.diff(centers)
    return (gaps.max() / page_width) > gap_ratio_thr

def assign_columns_kmeans(df: pd.DataFrame, max_cols: int = 3) -> pd.DataFrame:
    """Cluster words into columns using 1‑D KMeans and label them."""
    k = min(max_cols, len(df))
    km = KMeans(n_clusters=k, n_init="auto").fit(
        ((df.x0 + df.x1) / 2).to_numpy().reshape(-1, 1)
    )
    df["col"] = km.labels_
    order = df.groupby("col").x0.min().sort_values().index.tolist()
    df["col"] = df.col.map({old: new for new, old in enumerate(order)})
    return df

def rebuild_text_from_columns(df: pd.DataFrame, line_tol: int = 8) -> str:
    """Reconstruct reading order: left‑to‑right columns, then top‑to‑bottom."""
    lines = []
    for col in sorted(df.col.unique()):
        col_df = df[df.col == col].sort_values(["y0", "x0"])
        current, last_top = [], None
        for _, w in col_df.iterrows():
            if last_top is None or abs(w.y0 - last_top) <= line_tol:
                current.append(w.text)
            else:
                lines.append(" ".join(current))
                current = [w.text]
            last_top = w.y0
        if current:
            lines.append(" ".join(current))
    return "\n".join(lines)

def process_single_page(page_info: Tuple[int, Any], ocr_lang: str, ocr_dpi: int) -> Tuple[int, str]:
    """Process a single page and return (page_index, text)."""
    page_idx, page = page_info
    raw_text = page.get_text("text").strip()
    
    # Build a DataFrame of word boxes
    if raw_text:
        words = page.get_text("words")
        words_df = pd.DataFrame(
            words,
            columns=["x0", "y0", "x1", "y1", "text", "_b", "_l", "_w"]
        )[["x0", "y0", "x1", "y1", "text"]]
    else:
        words_df = ocr_page_words(page, dpi=ocr_dpi, lang=ocr_lang)
    
    # Determine layout and rebuild text accordingly
    if is_multicol(words_df, page.rect.width):
        words_df = assign_columns_kmeans(words_df, max_cols=3)
        page_text = rebuild_text_from_columns(words_df)
    else:
        page_text = " ".join(
            w.text for _, w in
            words_df.sort_values(["y0", "x0"]).iterrows()
        )
    
    return page_idx, page_text

def extract_pdf_content(pdf_path: str,
                       ocr_lang: str = "eng",
                       ocr_dpi: int = 350,
                       max_workers: int = 4) -> Dict[str, Any]:
    """Extract text from a PDF with optional OCR and column reordering.
    
    Parameters
    ----------
    pdf_path : str
        Path to the PDF file
    ocr_lang : str
        Language for OCR (default: "kor+eng")
    ocr_dpi : int
        DPI for OCR rendering (default: 350)
    max_workers : int
        Maximum number of parallel workers (default: 4)
    
    Returns
    -------
    Dict[str, Any]
        Extracted content with text, sections, etc.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_text: List[str] = [None] * total_pages
    
    # Prepare page data for parallel processing
    page_data = [(i, doc[i]) for i in range(total_pages)]
    
    # Process pages in parallel
    process_func = partial(process_single_page, ocr_lang=ocr_lang, ocr_dpi=ocr_dpi)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_page = {executor.submit(process_func, page_info): page_info[0] 
                         for page_info in page_data}
        
        # Collect results as they complete
        for future in as_completed(future_to_page):
            try:
                page_idx, page_text = future.result()
                pages_text[page_idx] = page_text
            except Exception as e:
                page_idx = future_to_page[future]
                print(f"[ERROR] Failed to process page {page_idx + 1}: {e}")
                pages_text[page_idx] = ""  # Empty text for failed pages

    toc = doc.get_toc(simple=True)  # using get_toc
    if toc:
        sections = build_sections_from_toc(toc, total_pages)
    else:
        sections = build_sections_from_layout(pdf_path)
        if not sections:
            sections = [{
                "title": f"Page {i + 1}",
                "start_page": i + 1,
                "end_page": i + 1,
                "method": "Page-based"
            } for i in range(total_pages)]

    return {
        "file_path": pdf_path,
        "toc": toc,
        "pages_text": pages_text,
        "sections": sections
    }

def save_extracted_content(content: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Directory for original PDFs (e.g., data/original)
    pdf_folder = os.path.join("data", "original")
    output_folder = os.path.join("data", "extracted")
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"[ERROR] No PDF files found in '{pdf_folder}'.")
        sys.exit(1)

    # If multiple PDFs exist, show the list and let the user choose
    if len(pdf_files) > 1:
        print("Multiple PDF files found:")
        for idx, fname in enumerate(pdf_files):
            print(f"{idx+1}. {fname}")
        selection = input("Select a file by number: ")
        try:
            selection_idx = int(selection) - 1
            if selection_idx < 0 or selection_idx >= len(pdf_files):
                print("Invalid selection.")
                sys.exit(1)
            selected_file = pdf_files[selection_idx]
        except ValueError:
            print("Invalid input.")
            sys.exit(1)
    else:
        selected_file = pdf_files[0]

    pdf_path = os.path.join(pdf_folder, selected_file)
    print(f"Processing file: {selected_file}")
    extracted_data = extract_pdf_content(pdf_path)

    base_name = os.path.splitext(selected_file)[0]
    output_json = os.path.join(output_folder, f"{base_name}.json")
    save_extracted_content(extracted_data, output_json)
    print(f"Processed {selected_file}: Found {len(extracted_data['sections'])} sections.")

    # Also save merged sections as sections.json for convenience
    sections_output = os.path.join(output_folder, "sections.json")
    with open(sections_output, 'w', encoding='utf-8') as f:
        json.dump(extracted_data["sections"], f, ensure_ascii=False, indent=2)
    print(f"Sections saved to {sections_output}")

    print("PDF Extraction Complete.")