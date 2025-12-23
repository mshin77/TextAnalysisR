# Lexical Diversity Analysis

Calculates lexical diversity metrics to measure vocabulary richness.
MTLD and MATTR are most stable and text-length independent.

## Usage

``` r
lexical_diversity_analysis(x, measures = "all", texts = NULL)
```

## Arguments

- x:

  A tokens object (preferred) or document-feature matrix from quanteda.
  For accurate MTLD calculation, pass a tokens object or provide the
  `texts` parameter. DFM input loses token order, which affects MTLD
  accuracy (McCarthy & Jarvis, 2010).

- measures:

  Character vector of measures to calculate. Options: "all", "MTLD"
  (recommended), "MATTR" (recommended), "MSTTR", "TTR", "CTTR", "Maas",
  "K", "D"

- texts:

  Optional character vector of original texts. Required for accurate
  MTLD when passing a DFM (since DFM loses token order). Also used for
  average sentence length.

## Value

A list with lexical_diversity (data frame) and summary_stats

## See also

Other lexical:
[`calculate_text_readability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_text_readability.md),
[`extract_keywords_keyness()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_keyness.md),
[`extract_keywords_tfidf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_tfidf.md),
[`lexical_frequency_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/lexical_frequency_analysis.md),
[`plot_keyness_keywords()`](https://mshin77.github.io/TextAnalysisR/reference/plot_keyness_keywords.md),
[`plot_keyword_comparison()`](https://mshin77.github.io/TextAnalysisR/reference/plot_keyword_comparison.md),
[`plot_lexical_diversity_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_lexical_diversity_distribution.md),
[`plot_readability_by_group()`](https://mshin77.github.io/TextAnalysisR/reference/plot_readability_by_group.md),
[`plot_readability_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_readability_distribution.md),
[`plot_tfidf_keywords()`](https://mshin77.github.io/TextAnalysisR/reference/plot_tfidf_keywords.md),
[`plot_top_readability_documents()`](https://mshin77.github.io/TextAnalysisR/reference/plot_top_readability_documents.md)

## Examples

``` r
if (FALSE) { # \dontrun{
data(SpecialEduTech)
texts <- SpecialEduTech$abstract[1:10]
corp <- quanteda::corpus(texts)
toks <- quanteda::tokens(corp)
# Preferred: pass tokens object for accurate MTLD
lex_div <- lexical_diversity_analysis(toks, texts = texts)
# Alternative: pass DFM with texts for MTLD accuracy
dfm_obj <- quanteda::dfm(toks)
lex_div <- lexical_diversity_analysis(dfm_obj, texts = texts)
print(lex_div)
} # }
```
