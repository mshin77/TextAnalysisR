# Extract PDF with Multimodal Analysis

Extract both text and visual content from PDFs, converting everything to
text for downstream analysis in your existing workflow.

## Usage

``` r
extract_pdf_multimodal(
  file_path,
  vision_provider = "ollama",
  vision_model = NULL,
  api_key = NULL,
  describe_images = TRUE,
  envname = "langgraph-env"
)
```

## Arguments

- file_path:

  Character string path to PDF file

- vision_provider:

  Character: "ollama" (local, default) or "openai" (cloud)

- vision_model:

  Character: Model name

  - For Ollama: "llava", "llava:13b", "bakllava"

  - For OpenAI: "gpt-4-vision-preview", "gpt-4o"

- api_key:

  Character: OpenAI API key (required if vision_provider="openai")

- describe_images:

  Logical: Convert images to text descriptions (default: TRUE)

- envname:

  Character: Python environment name (default: "langgraph-env")

## Value

List with:

- success: Logical

- combined_text: Character string with all content for text analysis

- text_content: List of text chunks

- image_descriptions: List of image descriptions

- num_images: Integer count of processed images

- vision_provider: Character indicating provider used

- message: Character status message

## Details

**Workflow Integration:**

1.  Extracts text using Marker (preserves equations, tables, structure)

2.  Detects images/charts/diagrams in PDF

3.  Uses vision LLM to describe visual content as text

4.  Merges text + descriptions → single text corpus

5.  Feed to existing text analysis pipeline

**Vision Provider Options:**

**Ollama (Default - Local & Free):**

- Privacy: Everything runs locally

- Cost: Free

- Setup: Requires Ollama installed + vision model pulled

- Models: llava, bakllava, llava-phi3

**OpenAI (Optional - Cloud):**

- Privacy: Data sent to OpenAI

- Cost: Paid (user's API key)

- Setup: Just provide API key

- Models: gpt-4-vision-preview, gpt-4o

## Examples

``` r
if (FALSE) { # \dontrun{
# Local analysis with Ollama (free, private)
result <- extract_pdf_multimodal("research_paper.pdf")

# Access combined text for analysis
text_for_analysis <- result$combined_text

# Use in existing workflow
corpus <- prep_texts(text_for_analysis)
topics <- fit_semantic_model(corpus, k = 5)

# Optional: Use OpenAI for better accuracy
result <- extract_pdf_multimodal(
  "paper.pdf",
  vision_provider = "openai",
  vision_model = "gpt-4o",
  api_key = Sys.getenv("OPENAI_API_KEY")
)
} # }
```
