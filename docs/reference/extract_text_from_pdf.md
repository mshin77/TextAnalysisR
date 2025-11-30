# Extract Text from PDF

Extracts text content from a PDF file using pdftools package.

## Usage

``` r
extract_text_from_pdf(file_path)
```

## Arguments

- file_path:

  Character string path to PDF file

## Value

Data frame with columns: page (integer), text (character) Returns NULL
if extraction fails or PDF is empty

## Details

Uses
[`pdftools::pdf_text()`](https://docs.ropensci.org/pdftools//reference/pdftools.html)
to extract text from each page. Preserves page structure and cleans
whitespace. Works best with text-based PDFs (not scanned images).

## Examples

``` r
if (FALSE) { # \dontrun{
pdf_path <- "path/to/document.pdf"
text_data <- extract_text_from_pdf(pdf_path)
head(text_data)
} # }
```
