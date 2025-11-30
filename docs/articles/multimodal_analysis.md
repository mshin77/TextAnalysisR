# Multimodal Analysis

Extract text from PDFs with charts, diagrams, and images using vision
AI.

## Setup

### Local (Ollama)

``` bash
# Install from https://ollama.ai
ollama pull llava
```

### Cloud (OpenAI)

``` r
Sys.setenv(OPENAI_API_KEY = "sk-...")
```

## Usage

``` r
library(TextAnalysisR)

# Extract PDF with images
result <- extract_pdf_multimodal(
  "document.pdf",
  vision_provider = "ollama"  # or "openai"
)

# Use in analysis
tokens <- prep_texts(result$combined_text)
```

## Smart Extraction

Auto-detects document type:

``` r
result <- extract_pdf_smart("paper.pdf", doc_type = "auto")
```

## Provider Comparison

| Provider | Cost    | Privacy | Accuracy |
|----------|---------|---------|----------|
| Ollama   | Free    | Local   | Good     |
| OpenAI   | Per use | Cloud   | Best     |
