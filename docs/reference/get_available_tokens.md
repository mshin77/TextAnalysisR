# Get Available Tokens with Fallback

Returns the first non-NULL tokens object from a priority fallback chain.
Useful when multiple token processing stages exist and you need the most
processed available version.

## Usage

``` r
get_available_tokens(
  final_tokens = NULL,
  processed_tokens = NULL,
  preprocessed_tokens = NULL,
  united_tbl = NULL
)
```

## Arguments

- final_tokens:

  Optional fully processed tokens (highest priority)

- processed_tokens:

  Optional partially processed tokens

- preprocessed_tokens:

  Optional early-stage preprocessed tokens

- united_tbl:

  Optional data frame with united_texts column (lowest priority, will be
  tokenized)

## Value

The first non-NULL tokens from the priority chain, or NULL if all are
NULL

## Details

Priority order (highest to lowest):

1.  final_tokens - Fully processed tokens

2.  processed_tokens - Partially processed tokens

3.  preprocessed_tokens - Early stage preprocessed tokens

4.  united_tbl - Raw text (will be tokenized if used)

## See also

Other preprocessing:
[`detect_multi_words()`](https://mshin77.github.io/TextAnalysisR/reference/detect_multi_words.md),
[`extract_named_entities()`](https://mshin77.github.io/TextAnalysisR/reference/extract_named_entities.md),
[`extract_pos_tags()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pos_tags.md),
[`get_available_dfm()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_dfm.md),
[`import_files()`](https://mshin77.github.io/TextAnalysisR/reference/import_files.md),
[`prep_texts()`](https://mshin77.github.io/TextAnalysisR/reference/prep_texts.md),
[`process_pdf_unified()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_unified.md),
[`unite_cols()`](https://mshin77.github.io/TextAnalysisR/reference/unite_cols.md)

## Examples

``` r
if (FALSE) { # \dontrun{
tokens <- get_available_tokens(
  final_tokens = my_final_tokens,
  processed_tokens = my_processed_tokens
)
} # }
```
