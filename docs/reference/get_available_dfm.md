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

## Examples

``` r
if (FALSE) { # \dontrun{
dfm1 <- quanteda::dfm(quanteda::tokens("assistive technology supports learning"))
result <- get_available_dfm(dfm_init = dfm1)
} # }
```
