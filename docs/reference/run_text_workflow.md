# Complete Text Mining Workflow

This function provides a complete text mining workflow that follows the
same sequence as the Shiny application: file processing → text uniting →
preprocessing → DFM creation → analysis. It serves as a convenience
function for users who want to execute the entire pipeline
programmatically.

## Usage

``` r
run_text_workflow(
  dataset_choice,
  file_info = NULL,
  text_input = NULL,
  listed_vars,
  min_char = 2,
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_numbers = TRUE,
  remove_url = TRUE,
  detect_compounds = FALSE,
  compound_size = 2:3,
  compound_min_count = 2,
  verbose = TRUE
)
```

## Arguments

- dataset_choice:

  A character string indicating the dataset choice: "Upload an Example
  Dataset", "Upload Your File", "Copy and Paste Text".

- file_info:

  A data frame containing file information (for file upload).

- text_input:

  A character string containing text input (for copy-paste).

- listed_vars:

  A character vector of column names to unite into text.

- min_char:

  The minimum number of characters for tokens (default: 2).

- remove_punct:

  Logical; remove punctuation (default: TRUE).

- remove_symbols:

  Logical; remove symbols (default: TRUE).

- remove_numbers:

  Logical; remove numbers (default: TRUE).

- remove_url:

  Logical; remove URLs (default: TRUE).

- detect_compounds:

  Logical; detect multi-word expressions (default: FALSE).

- compound_size:

  Size range for compound detection (default: 2:3).

- compound_min_count:

  Minimum count for compounds (default: 2).

- verbose:

  Logical; print progress messages (default: TRUE).

## Value

A list containing processed data, tokens, DFM, and metadata.

## Examples

``` r
if (interactive()) {
  # Using example dataset
  workflow_result <- TextAnalysisR::run_text_workflow(
    dataset_choice = "Upload an Example Dataset",
    listed_vars = c("title", "keyword", "abstract")
  )

  # Using file upload
  file_info <- data.frame(filepath = "path/to/your/file.xlsx")
  workflow_result <- TextAnalysisR::run_text_workflow(
    dataset_choice = "Upload Your File",
    file_info = file_info,
    listed_vars = c("column1", "column2")
  )

  # Using copy-paste text
  workflow_result <- TextAnalysisR::run_text_workflow(
    dataset_choice = "Copy and Paste Text",
    text_input = "Your text content here",
    listed_vars = "text"
  )
}
```
