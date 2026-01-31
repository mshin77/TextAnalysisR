# Multimodal Analysis

Extract text from PDFs with charts, diagrams, and images using vision
AI. R-native pipeline – no Python required.

## How It Works

1.  Extracts text from each page using
    [`pdftools::pdf_text()`](https://docs.ropensci.org/pdftools//reference/pdftools.html)
    (R-native)
2.  Renders each page as a PNG image via
    [`pdftools::pdf_render_page()`](https://docs.ropensci.org/pdftools//reference/pdf_render_page.html)
3.  Identifies sparse-text pages (\< 500 characters) that likely contain
    figures
4.  Sends only those pages to a vision LLM for description
5.  Merges extracted text + image descriptions into a single text corpus

## Setup

### Local (Ollama)

``` bash
# Install from https://ollama.com
ollama pull llava          # General purpose (default)
ollama pull bakllava       # Mistral-based alternative
ollama pull llava-phi3     # Lightweight option
```

### Cloud (OpenAI / Gemini)

``` r
Sys.setenv(OPENAI_API_KEY = "sk-...")
Sys.setenv(GEMINI_API_KEY = "your-gemini-key")
```

## Usage

``` r
library(TextAnalysisR)

# Extract PDF with vision AI (default: Ollama)
result <- extract_pdf_multimodal(
  "document.pdf",
  vision_provider = "ollama"  # or "openai" or "gemini"
)

# Use in analysis
tokens <- prep_texts(result$combined_text)
```

### Gemini Example

``` r
result <- extract_pdf_multimodal(
  "paper.pdf",
  vision_provider = "gemini",
  api_key = Sys.getenv("GEMINI_API_KEY")
)
```

### Describe Individual Images

``` r
description <- describe_image(
  image_base64,
  provider = "openai",
  api_key = Sys.getenv("OPENAI_API_KEY")
)
```

## Unified PDF Pipeline

[`process_pdf_unified()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_unified.md)
provides automatic fallback:

1.  **Multimodal** (pdftools + vision LLM) – extracts text and describes
    visual content
2.  **Text-only** (pdftools) – fallback if no vision provider is
    available

``` r
result <- process_pdf_unified("paper.pdf", vision_provider = "gemini")
```

## Provider Comparison

| Provider | Cost    | Privacy | Accuracy | Setup                       |
|----------|---------|---------|----------|-----------------------------|
| Ollama   | Free    | Local   | Good     | Install Ollama + pull model |
| OpenAI   | Per use | Cloud   | Best     | API key                     |
| Gemini   | Per use | Cloud   | Best     | API key                     |
