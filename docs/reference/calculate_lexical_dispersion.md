# Calculate Lexical Dispersion

Computes lexical dispersion data for specified terms across a corpus.
Shows where terms appear within each document, useful for understanding
term distribution patterns.

## Usage

``` r
calculate_lexical_dispersion(
  tokens_object,
  terms,
  scale = c("relative", "absolute")
)
```

## Arguments

- tokens_object:

  A quanteda tokens object

- terms:

  Character vector of terms to analyze

- scale:

  Character, "relative" (0-1 normalized) or "absolute" (token position)

## Value

Data frame with columns:

- doc_id: Document identifier

- term: The search term

- position: Position in document (relative or absolute)

- doc_length: Total tokens in document

## Examples

``` r
# \donttest{
tokens <- quanteda::tokens(TextAnalysisR::SpecialEduTech$abstract[1:5])
dispersion <- calculate_lexical_dispersion(tokens, c("learning", "instruction"))
# }
```
