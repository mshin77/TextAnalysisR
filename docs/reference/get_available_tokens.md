# Get Available Tokens with Fallback

Returns the first non-NULL tokens object from a priority fallback chain.
Useful when multiple token processing stages exist and the most
processed available version is needed.

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

## Examples

``` r
if (interactive()) {
  toks <- quanteda::tokens("assistive technology supports learning")
  result <- get_available_tokens(processed_tokens = toks)
}
```
