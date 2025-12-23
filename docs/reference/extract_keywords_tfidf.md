# Extract Keywords Using TF-IDF

Extracts top keywords from a document-feature matrix using TF-IDF
weighting.

## Usage

``` r
extract_keywords_tfidf(dfm, top_n = 20, normalize = FALSE)
```

## Arguments

- dfm:

  A quanteda dfm object

- top_n:

  Number of top keywords to extract (default: 20)

- normalize:

  Logical, whether to normalize TF-IDF scores to 0-1 range (default:
  FALSE)

## Value

Data frame with columns: Keyword, TF_IDF_Score, Frequency

## See also

Other lexical:
[`calculate_text_readability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_text_readability.md),
[`extract_keywords_keyness()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_keyness.md),
[`lexical_diversity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/lexical_diversity_analysis.md),
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
library(quanteda)
corp <- corpus(c("text analysis", "data mining", "text mining"))
dfm_obj <- dfm(tokens(corp))
keywords <- extract_keywords_tfidf(dfm_obj, top_n = 5)
print(keywords)
} # }
```
