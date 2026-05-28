# Multimodal Analysis

``` r

library(TextAnalysisR)

# Text-only fallback runs without any vision provider.
# prep_texts handles the cleaned text that PDF extraction yields.
sample_text <- c(
  "Figure 1 shows the distribution of student outcomes.",
  "Table 2 reports the effect sizes for each intervention."
)
toks <- prep_texts(
  data.frame(united_texts = sample_text),
  text_field = "united_texts"
)
quanteda::ntoken(toks)
```

    ## text1 text2 
    ##     7     8

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

### Cloud (OpenAI / Gemini)

``` r

Sys.setenv(OPENAI_API_KEY = "sk-...")
Sys.setenv(GEMINI_API_KEY = "<gemini-api-key>")
```

## Usage

``` r

library(TextAnalysisR)

# Extract PDF with vision AI
result <- extract_pdf_multimodal(
  "document.pdf",
  vision_provider = "gemini"  # or "openai"
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

| Provider | Cost | Privacy | Accuracy | Setup |
|----|----|----|----|----|
| OpenAI | Per use | Cloud | Best | API key |
| Gemini | Free on hosted app (Google Cloud Research); otherwise per use | Cloud | Best | API key |
