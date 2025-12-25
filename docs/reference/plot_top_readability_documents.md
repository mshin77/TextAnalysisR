# Plot Top Documents by Readability

Creates a bar plot of documents ranked by readability metric.

## Usage

``` r
plot_top_readability_documents(
  readability_data,
  metric,
  top_n = 15,
  order = "highest",
  title = NULL
)
```

## Arguments

- readability_data:

  Data frame from calculate_text_readability()

- metric:

  Metric to plot

- top_n:

  Number of documents to show (default: 15)

- order:

  Direction: "highest" or "lowest" (default: "highest")

- title:

  Plot title (default: auto-generated)

## Value

A plotly bar chart

## See also

Other lexical:
[`calculate_text_readability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_text_readability.md),
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
[`render_displacy_dep()`](https://mshin77.github.io/TextAnalysisR/reference/render_displacy_dep.md),
[`render_displacy_ent()`](https://mshin77.github.io/TextAnalysisR/reference/render_displacy_ent.md),
[`summarize_morphology()`](https://mshin77.github.io/TextAnalysisR/reference/summarize_morphology.md)
