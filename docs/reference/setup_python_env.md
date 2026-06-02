# Setup Python Environment

Sets up a tiered Python virtual environment.

## Usage

``` r
setup_python_env(envname = "textanalysisr-env", tier = "core", force = FALSE)
```

## Arguments

- envname:

  Character string name for the virtual environment (default:
  "textanalysisr-env")

- tier:

  Which feature tier to install. One or more of: `"core"` (spacy +
  pdfplumber, ~200 MB; default), `"embeddings"` (adds
  sentence-transformers + transformers + torch, ~1 GB), `"topics"` (adds
  BERTopic + UMAP + HDBSCAN, ~300 MB on top of embeddings).

- force:

  Logical, whether to recreate environment if it exists (default: FALSE)

## Value

Invisible TRUE if successful, stops with error message if failed

## Details

Default `tier = "core"` keeps the install light – spaCy NLP and PDF text
extraction only. Add `"embeddings"` for sentence-transformers-based
similarity/sentiment, and `"topics"` for BERTopic.

The virtual environment is isolated; system Python is not modified.

## Examples

``` r
if (interactive()) {
  setup_python_env()                              # core only (~200 MB)
  setup_python_env(tier = c("core", "embeddings"))  # +1 GB
  setup_python_env(tier = c("core", "embeddings", "topics"))  # full stack
}
```
