# Calculate Text Readability

Calculates multiple readability metrics for texts including Flesch
Reading Ease, Flesch-Kincaid Grade Level, Gunning FOG index, and others.
Optionally includes lexical diversity metrics and sentence statistics.

## Usage

``` r
calculate_text_readability(
  texts,
  metrics = c("flesch", "flesch_kincaid", "gunning_fog"),
  include_lexical_diversity = TRUE,
  include_sentence_stats = TRUE,
  dfm_for_lexdiv = NULL,
  doc_names = NULL
)
```

## Arguments

- texts:

  Character vector of texts to analyze

- metrics:

  Character vector of readability metrics to calculate. Options:
  "flesch", "flesch_kincaid", "gunning_fog", "smog", "ari",
  "coleman_liau"

- include_lexical_diversity:

  Logical, include TTR and MTLD (default: TRUE)

- include_sentence_stats:

  Logical, include average sentence length (default: TRUE)

- dfm_for_lexdiv:

  Optional pre-computed DFM for lexical diversity calculation

- doc_names:

  Optional character vector of document names

## Value

A data frame with document names and readability scores

## See also

Other lexical:
[`clear_lexdiv_cache()`](https://mshin77.github.io/TextAnalysisR/reference/clear_lexdiv_cache.md),
[`detect_multi_words()`](https://mshin77.github.io/TextAnalysisR/reference/detect_multi_words.md),
[`extract_keywords_keyness()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_keyness.md),
[`extract_keywords_tfidf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_tfidf.md),
[`extract_morphology()`](https://mshin77.github.io/TextAnalysisR/reference/extract_morphology.md),
[`extract_named_entities()`](https://mshin77.github.io/TextAnalysisR/reference/extract_named_entities.md),
[`extract_pos_tags()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pos_tags.md),
[`lexical_analysis`](https://mshin77.github.io/TextAnalysisR/reference/lexical_analysis.md),
[`lexical_diversity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/lexical_diversity_analysis.md),
[`lexical_frequency_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/lexical_frequency_analysis.md),
[`plot_keyness_keywords()`](https://mshin77.github.io/TextAnalysisR/reference/plot_keyness_keywords.md),
[`plot_keyword_comparison()`](https://mshin77.github.io/TextAnalysisR/reference/plot_keyword_comparison.md),
[`plot_lexical_diversity_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_lexical_diversity_distribution.md),
[`plot_morphology_feature()`](https://mshin77.github.io/TextAnalysisR/reference/plot_morphology_feature.md),
[`plot_readability_by_group()`](https://mshin77.github.io/TextAnalysisR/reference/plot_readability_by_group.md),
[`plot_readability_distribution()`](https://mshin77.github.io/TextAnalysisR/reference/plot_readability_distribution.md),
[`plot_tfidf_keywords()`](https://mshin77.github.io/TextAnalysisR/reference/plot_tfidf_keywords.md),
[`plot_top_readability_documents()`](https://mshin77.github.io/TextAnalysisR/reference/plot_top_readability_documents.md),
[`render_displacy_dep()`](https://mshin77.github.io/TextAnalysisR/reference/render_displacy_dep.md),
[`render_displacy_ent()`](https://mshin77.github.io/TextAnalysisR/reference/render_displacy_ent.md),
[`summarize_morphology()`](https://mshin77.github.io/TextAnalysisR/reference/summarize_morphology.md)

## Examples

``` r
if (FALSE) { # \dontrun{
texts <- c(
  "This is simple text.",
  "This sentence contains more complex vocabulary and structure."
)
readability <- calculate_text_readability(texts)
print(readability)
} # }
```
