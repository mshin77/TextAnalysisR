import pdfplumber
import pandas as pd
from typing import Dict, List, Optional, Tuple


def extract_text_from_pdf(file_path: str) -> Dict:
    """
    Extract text from PDF using pdfplumber.

    Args:
        file_path: Path to PDF file

    Returns:
        Dictionary with:
            - success: bool
            - data: List of dicts with page and text
            - total_pages: int
            - message: str
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                return {
                    "success": False,
                    "data": None,
                    "total_pages": 0,
                    "message": "PDF file is empty"
                }

            pages_data = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_data.append({
                        "page": i + 1,
                        "text": text.strip()
                    })

            if not pages_data:
                return {
                    "success": False,
                    "data": None,
                    "total_pages": len(pdf.pages),
                    "message": "PDF contains no readable text"
                }

            return {
                "success": True,
                "data": pages_data,
                "total_pages": len(pdf.pages),
                "message": f"Successfully extracted text from {len(pages_data)} pages"
            }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "total_pages": 0,
            "message": f"Error extracting text: {str(e)}"
        }


def extract_tables_from_pdf(file_path: str, pages: Optional[List[int]] = None) -> Dict:
    """
    Extract tables from PDF using pdfplumber.

    Args:
        file_path: Path to PDF file
        pages: Optional list of page numbers to process (1-indexed)

    Returns:
        Dictionary with:
            - success: bool
            - data: Dict representation of DataFrame
            - num_tables: int
            - message: str
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            all_tables = []
            pages_to_process = pages if pages else range(1, len(pdf.pages) + 1)

            for page_num in pages_to_process:
                if page_num > len(pdf.pages):
                    continue

                page = pdf.pages[page_num - 1]
                tables = page.extract_tables()

                for table in tables:
                    if table and len(table) > 1:
                        all_tables.append(table)

            if not all_tables:
                return {
                    "success": False,
                    "data": None,
                    "num_tables": 0,
                    "message": "No tables detected in PDF"
                }

            first_table = all_tables[0]

            if not first_table or len(first_table) == 0:
                return {
                    "success": False,
                    "data": None,
                    "num_tables": len(all_tables),
                    "message": "Extracted table is empty"
                }

            df = pd.DataFrame(first_table[1:], columns=first_table[0])

            df.columns = [str(col) if col else f"Column_{i}"
                         for i, col in enumerate(df.columns)]

            df = df.dropna(how='all')

            if df.empty:
                return {
                    "success": False,
                    "data": None,
                    "num_tables": len(all_tables),
                    "message": "Table extraction resulted in empty data"
                }

            return {
                "success": True,
                "data": df.to_dict('list'),
                "num_tables": len(all_tables),
                "num_rows": len(df),
                "num_cols": len(df.columns),
                "message": f"Successfully extracted {len(df)} rows from PDF table"
            }

    except Exception as e:
        return {
            "success": False,
            "data": None,
            "num_tables": 0,
            "message": f"Error extracting tables: {str(e)}"
        }


def detect_pdf_content_type(file_path: str) -> Dict:
    """
    Detect if PDF contains primarily tables or text.

    Args:
        file_path: Path to PDF file

    Returns:
        Dictionary with:
            - content_type: "tabular", "text", or "unknown"
            - confidence: float (0-1)
            - message: str
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                return {
                    "content_type": "unknown",
                    "confidence": 0.0,
                    "message": "PDF is empty"
                }

            pages_to_sample = min(3, len(pdf.pages))
            table_count = 0
            text_length = 0

            for i in range(pages_to_sample):
                page = pdf.pages[i]

                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        if table and len(table) > 1 and len(table[0]) > 1:
                            table_count += 1

                text = page.extract_text()
                if text:
                    text_length += len(text.strip())

            if table_count > 0:
                confidence = min(0.9, 0.5 + (table_count * 0.2))
                return {
                    "content_type": "tabular",
                    "confidence": confidence,
                    "message": f"Detected {table_count} tables in first {pages_to_sample} pages"
                }

            if text_length > 100:
                return {
                    "content_type": "text",
                    "confidence": 0.8,
                    "message": f"Detected {text_length} characters of text"
                }

            return {
                "content_type": "unknown",
                "confidence": 0.0,
                "message": "Could not determine content type"
            }

    except Exception as e:
        return {
            "content_type": "unknown",
            "confidence": 0.0,
            "message": f"Error detecting content type: {str(e)}"
        }


def process_pdf_file(file_path: str, content_type: str = "auto") -> Dict:
    """
    Main function to process PDF files with automatic content detection.

    Args:
        file_path: Path to PDF file
        content_type: "auto", "text", or "tabular"

    Returns:
        Dictionary with:
            - success: bool
            - type: "text", "tabular", or "error"
            - data: Extracted data
            - message: str
    """
    try:
        if content_type == "auto":
            detection = detect_pdf_content_type(file_path)
            detected_type = detection["content_type"]
        else:
            detected_type = content_type

        if detected_type == "tabular":
            result = extract_tables_from_pdf(file_path)
            if result["success"]:
                return {
                    "success": True,
                    "type": "tabular",
                    "data": result["data"],
                    "message": result["message"]
                }
            else:
                text_result = extract_text_from_pdf(file_path)
                if text_result["success"]:
                    return {
                        "success": True,
                        "type": "text",
                        "data": text_result["data"],
                        "message": f"{text_result['message']} (table extraction failed)"
                    }
                return {
                    "success": False,
                    "type": "error",
                    "data": None,
                    "message": "Could not extract tables or text"
                }

        text_result = extract_text_from_pdf(file_path)
        if text_result["success"]:
            return {
                "success": True,
                "type": "text",
                "data": text_result["data"],
                "message": text_result["message"]
            }

        return {
            "success": False,
            "type": "error",
            "data": None,
            "message": text_result["message"]
        }

    except Exception as e:
        return {
            "success": False,
            "type": "error",
            "data": None,
            "message": f"Error processing PDF: {str(e)}"
        }
