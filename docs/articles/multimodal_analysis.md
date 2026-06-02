# Multimodal Analysis

``` r

library(TextAnalysisR)

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

## Functions

[`process_pdf_unified()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_unified.md)
runs the full pipeline with automatic fallback:

1.  **Multimodal** (pdftools + vision LLM) – extracts text and describes
    visual content
2.  **Text-only** (pdftools) – fallback when no vision provider is set

[`describe_image()`](https://mshin77.github.io/TextAnalysisR/reference/describe_image.md)
describes a single base64-encoded PNG. Both require a vision-provider
API key (OpenAI/Gemini) and network access; see their reference pages
for usage.

## Provider Comparison

| Provider | Cost | Privacy | Accuracy | Setup |
|----|----|----|----|----|
| OpenAI | Per use | Cloud | Best | API key |
| Gemini | Free on hosted app (Google Cloud Research); otherwise per use | Cloud | Best | API key |
