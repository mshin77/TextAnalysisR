# Lemmatize Tokens with Batch Processing

Converts tokens to their lemmatized forms using spaCy, with batch
processing to handle large document collections without timeout issues.

## Usage

``` r
lemmatize_tokens(
  tokens,
  batch_size = 50,
  model = "en_core_web_sm",
  verbose = TRUE
)
```

## Arguments

- tokens:

  A quanteda tokens object to lemmatize.

- batch_size:

  Integer; number of documents to process per batch (default: 50).

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

- verbose:

  Logical; print progress messages (default: TRUE).

## Value

A quanteda tokens object containing lemmatized tokens.

## Details

Uses spaCy for linguistic lemmatization producing proper dictionary
forms (e.g., "studies" -\> "study", "better" -\> "good"). Batch
processing prevents timeout errors with large document collections.

## Examples

``` r
if (interactive()) {
tokens <- quanteda::tokens(c("The studies showed better results"))
lemmatized <- lemmatize_tokens(tokens, batch_size = 50)
}
```
