# Preprocess Text Data

Preprocesses text data following the complete workflow implemented in
the Shiny application:

- Constructing a corpus from united texts

- Tokenizing text into words with configurable options

- Converting to lowercase with acronym preservation option

- Applying character length filtering

- Optional multi-word expression detection and compound term creation

- Stopword removal and lemmatization capabilities

This function serves as the foundation for all subsequent text analysis
workflows.

## Usage

``` r
prep_texts(
  united_tbl,
  text_field = "united_texts",
  min_char = 2,
  lowercase = TRUE,
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_numbers = TRUE,
  remove_url = TRUE,
  remove_separators = TRUE,
  split_hyphens = TRUE,
  split_tags = TRUE,
  include_docvars = TRUE,
  keep_acronyms = FALSE,
  padding = FALSE,
  verbose = FALSE,
  ...
)
```

## Arguments

- united_tbl:

  A data frame that contains text data.

- text_field:

  The name of the column that contains the text data.

- min_char:

  The minimum number of characters for a token to be included (default:
  2).

- lowercase:

  Logical; convert all tokens to lowercase (default: TRUE). Recommended
  for most text analysis tasks.

- remove_punct:

  Logical; remove punctuation from the text (default: TRUE).

- remove_symbols:

  Logical; remove symbols from the text (default: TRUE).

- remove_numbers:

  Logical; remove numbers from the text (default: TRUE).

- remove_url:

  Logical; remove URLs from the text (default: TRUE).

- remove_separators:

  Logical; remove separators from the text (default: TRUE).

- split_hyphens:

  Logical; split hyphenated words into separate tokens (default: TRUE).

- split_tags:

  Logical; split tags into separate tokens (default: TRUE).

- include_docvars:

  Logical; include document variables in the tokens object (default:
  TRUE).

- keep_acronyms:

  Logical; keep acronyms in the text (default: FALSE).

- padding:

  Logical; add padding to the tokens object (default: FALSE).

- verbose:

  Logical; print verbose output (default: FALSE).

- ...:

  Additional arguments passed to
  [`quanteda::tokens`](https://quanteda.io/reference/tokens.html).

## Value

A `tokens` object that contains the preprocessed text data.

## Examples

``` r
if (interactive()) {
mydata <- TextAnalysisR::SpecialEduTech

united_tbl <- TextAnalysisR::unite_cols(
  mydata,
  listed_vars = c("title", "keyword", "abstract")
)

tokens <- TextAnalysisR::prep_texts(united_tbl,
                                         text_field = "united_texts",
                                         min_char = 2,
                                         lowercase = TRUE,
                                         remove_punct = TRUE,
                                         remove_symbols = TRUE,
                                         remove_numbers = TRUE,
                                         remove_url = TRUE,
                                         remove_separators = TRUE,
                                         split_hyphens = TRUE,
                                         split_tags = TRUE,
                                         include_docvars = TRUE,
                                         keep_acronyms = FALSE,
                                         padding = FALSE,
                                         verbose = FALSE)
print(tokens)
}
```
