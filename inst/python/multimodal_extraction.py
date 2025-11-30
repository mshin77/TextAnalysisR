"""
Multimodal PDF Extraction with Vision LLMs

Extracts both text and visual content (charts, diagrams, images) from PDFs,
converting everything to text for downstream text analysis.
"""

import base64
import io
import json
from typing import Dict, List, Optional
from pathlib import Path

try:
    from pdf2image import convert_from_path
    import marker
    from marker.convert import convert_single_pdf
    HAS_MARKER = True
except ImportError:
    HAS_MARKER = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def extract_pdf_with_images(
    file_path: str,
    vision_provider: str = "ollama",
    vision_model: str = "llava",
    api_key: Optional[str] = None,
    describe_images: bool = True
) -> Dict:
    """
    Extract both text and images from PDF, converting images to descriptions.

    Args:
        file_path: Path to PDF file
        vision_provider: "ollama" (local, free) or "openai" (cloud, API key required)
        vision_model: Model name - "llava" for Ollama, "gpt-4-vision-preview" for OpenAI
        api_key: OpenAI API key (required if vision_provider="openai")
        describe_images: If True, convert images to text descriptions

    Returns:
        Dictionary with:
            - success: bool
            - text_content: List of text chunks
            - image_descriptions: List of image descriptions
            - combined_text: All text merged for analysis
            - total_pages: int
            - num_images: int
            - message: str
    """
    try:
        if not HAS_MARKER:
            return {
                "success": False,
                "message": "marker-pdf not installed. Install: pip install marker-pdf"
            }

        # Step 1: Extract text and layout using Marker
        markdown_text, images, metadata = convert_single_pdf(file_path)

        text_chunks = [markdown_text]
        image_descriptions = []

        # Step 2: Process images if requested
        if describe_images and images:
            if vision_provider == "ollama":
                image_descriptions = _describe_images_ollama(images, vision_model)
            elif vision_provider == "openai":
                if not api_key:
                    return {
                        "success": False,
                        "message": "OpenAI API key required for vision_provider='openai'"
                    }
                image_descriptions = _describe_images_openai(images, vision_model, api_key)
            else:
                return {
                    "success": False,
                    "message": f"Unknown vision_provider: {vision_provider}"
                }

        # Step 3: Combine text and image descriptions
        combined_text = markdown_text
        if image_descriptions:
            combined_text += "\n\n## Visual Content Descriptions\n\n"
            combined_text += "\n\n".join(image_descriptions)

        return {
            "success": True,
            "text_content": text_chunks,
            "image_descriptions": image_descriptions,
            "combined_text": combined_text,
            "total_pages": metadata.get("pages", 0),
            "num_images": len(images),
            "vision_provider": vision_provider,
            "message": f"Extracted text and {len(image_descriptions)} image descriptions"
        }

    except Exception as e:
        return {
            "success": False,
            "text_content": [],
            "image_descriptions": [],
            "combined_text": "",
            "message": f"Error: {str(e)}"
        }


def _describe_images_ollama(images: Dict, model: str = "llava") -> List[str]:
    """
    Describe images using local Ollama vision model.

    Args:
        images: Dictionary of image data from Marker
        model: Ollama vision model name

    Returns:
        List of text descriptions
    """
    if not HAS_REQUESTS:
        return []

    descriptions = []

    for img_name, img_data in images.items():
        try:
            # Convert image to base64
            if HAS_PIL:
                img = Image.open(io.BytesIO(img_data))
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
            else:
                img_base64 = base64.b64encode(img_data).decode()

            # Call Ollama API
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": "Describe this image in detail, focusing on any charts, diagrams, tables, or textual content. Extract any visible text.",
                    "images": [img_base64],
                    "stream": False
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                description = result.get("response", "")
                if description:
                    descriptions.append(f"**Image: {img_name}**\n{description}")

        except Exception as e:
            descriptions.append(f"**Image: {img_name}**\nError extracting: {str(e)}")

    return descriptions


def _describe_images_openai(
    images: Dict,
    model: str = "gpt-4-vision-preview",
    api_key: str = None
) -> List[str]:
    """
    Describe images using OpenAI Vision API.

    Args:
        images: Dictionary of image data from Marker
        model: OpenAI vision model name
        api_key: OpenAI API key

    Returns:
        List of text descriptions
    """
    if not HAS_REQUESTS:
        return []

    descriptions = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    for img_name, img_data in images.items():
        try:
            # Convert image to base64
            img_base64 = base64.b64encode(img_data).decode()

            # Call OpenAI API
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this image in detail, focusing on any charts, diagrams, tables, or textual content. Extract any visible text."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 500
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                description = result["choices"][0]["message"]["content"]
                if description:
                    descriptions.append(f"**Image: {img_name}**\n{description}")

        except Exception as e:
            descriptions.append(f"**Image: {img_name}**\nError extracting: {str(e)}")

    return descriptions


def extract_with_nougat(file_path: str) -> Dict:
    """
    Extract academic PDFs with equations using Nougat.
    Best for scientific papers with mathematical notation.

    Args:
        file_path: Path to PDF file

    Returns:
        Dictionary with extracted markdown including LaTeX equations
    """
    try:
        from nougat import NougatModel
        from nougat.utils.checkpoint import get_checkpoint

        model = NougatModel.from_pretrained('facebook/nougat-base')
        model.eval()

        # Process PDF
        predictions = model.inference(pdf_path=file_path)

        return {
            "success": True,
            "combined_text": predictions,
            "format": "markdown",
            "has_equations": True,
            "message": "Extracted with equation support (Nougat)"
        }

    except ImportError:
        return {
            "success": False,
            "message": "nougat-ocr not installed. Install: pip install nougat-ocr"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Nougat error: {str(e)}"
        }


def extract_pdf_smart(
    file_path: str,
    doc_type: str = "auto",
    vision_provider: str = "ollama",
    vision_model: str = "llava",
    api_key: Optional[str] = None
) -> Dict:
    """
    Smart PDF extraction with automatic method selection.

    Strategy:
    - Academic papers → Nougat (equations)
    - Documents with images → Multimodal extraction
    - General documents → Marker

    Args:
        file_path: Path to PDF file
        doc_type: "auto", "academic", or "general"
        vision_provider: "ollama" or "openai"
        vision_model: Model name for vision analysis
        api_key: API key for cloud providers

    Returns:
        Dictionary with extracted content optimized for text analysis
    """
    try:
        # Auto-detect document type if needed
        if doc_type == "auto":
            doc_type = _detect_document_type(file_path)

        # Choose extraction method
        if doc_type == "academic":
            # Try Nougat first for equations
            result = extract_with_nougat(file_path)
            if result["success"]:
                return result

        # Default: Multimodal extraction with Marker
        result = extract_pdf_with_images(
            file_path=file_path,
            vision_provider=vision_provider,
            vision_model=vision_model,
            api_key=api_key,
            describe_images=True
        )

        return result

    except Exception as e:
        return {
            "success": False,
            "message": f"Error: {str(e)}"
        }


def _detect_document_type(file_path: str) -> str:
    """Detect if document is academic or general"""
    try:
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                return "general"

            first_page = pdf.pages[0].extract_text() or ""

            # Academic indicators
            academic_keywords = ['abstract', 'references', 'arxiv', 'doi:', 'et al', 'equation']
            if any(kw in first_page.lower() for kw in academic_keywords):
                return "academic"

        return "general"

    except Exception:
        return "general"
