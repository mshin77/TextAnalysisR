# Get Available Document-Feature Matrix with Fallback

Returns the first non-NULL DFM from a priority fallback chain. Useful
when multiple DFM processing stages exist and you need the most
processed available version.

## Usage

``` r
get_available_dfm(
  dfm_lemma = NULL,
  dfm_outcome = NULL,
  dfm_final = NULL,
  dfm_init = NULL
)
```

## Arguments

- dfm_lemma:

  Optional lemmatized DFM (highest priority)

- dfm_outcome:

  Optional preprocessed DFM (medium priority)

- dfm_final:

  Optional final processed DFM (medium-low priority)

- dfm_init:

  Optional initial DFM (lowest priority)

## Value

The first non-NULL DFM from the priority chain, or NULL if all are NULL

## Details

Priority order (highest to lowest):

1.  dfm_lemma - Lemmatized tokens (most processed)

2.  dfm_outcome - Preprocessed tokens

3.  dfm_final - Final processed version

4.  dfm_init - Initial unprocessed tokens

## See also

Other preprocessing:
[`detect_multi_words()`](https://mshin77.github.io/TextAnalysisR/reference/detect_multi_words.md),
[`extract_named_entities()`](https://mshin77.github.io/TextAnalysisR/reference/extract_named_entities.md),
[`extract_pos_tags()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pos_tags.md),
[`get_available_tokens()`](https://mshin77.github.io/TextAnalysisR/reference/get_available_tokens.md),
[`import_files()`](https://mshin77.github.io/TextAnalysisR/reference/import_files.md),
[`prep_texts()`](https://mshin77.github.io/TextAnalysisR/reference/prep_texts.md),
[`process_pdf_unified()`](https://mshin77.github.io/TextAnalysisR/reference/process_pdf_unified.md),
[`unite_cols()`](https://mshin77.github.io/TextAnalysisR/reference/unite_cols.md)

## Examples

``` r
if (FALSE) { # \dontrun{
dfm1 <- quanteda::dfm(quanteda::tokens("assistive technology supports learning"))
result <- get_available_dfm(dfm_init = dfm1)
} # }
```
