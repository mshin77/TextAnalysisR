# Lexical Diversity Analysis

Calculates lexical diversity metrics to measure vocabulary richness.
MTLD and MATTR are most stable and text-length independent.

## Usage

``` r
lexical_diversity_analysis(dfm_object, measures = "all")
```

## Arguments

- dfm_object:

  A document-feature matrix from quanteda

- measures:

  Character vector of measures to calculate. Options: "all", "MTLD"
  (recommended), "MATTR" (recommended), "MSTTR", "TTR", "CTTR", "Maas",
  "K", "D"

## Value

A list with lexical_diversity (data frame) and summary_stats

## Examples

``` r
if (FALSE) { # \dontrun{
data(SpecialEduTech)
texts <- SpecialEduTech$abstract[1:10]
corp <- quanteda::corpus(texts)
toks <- quanteda::tokens(corp)
dfm_obj <- quanteda::dfm(toks)
lex_div <- lexical_diversity_analysis(dfm_obj)
print(lex_div)
} # }
```
