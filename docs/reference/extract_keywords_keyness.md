# Extract Keywords Using Statistical Keyness

Extracts distinctive keywords by comparing document groups using
log-likelihood ratio (G-squared).

## Usage

``` r
extract_keywords_keyness(dfm, target, top_n = 20, measure = "lr")
```

## Arguments

- dfm:

  A quanteda dfm object

- target:

  Target document indices or logical vector

- top_n:

  Number of top keywords to extract (default: 20)

- measure:

  Keyness measure: "lr" (log-likelihood) or "chi2" (default: "lr")

## Value

Data frame with columns: Keyword, Keyness_Score

## See also

Other lexical:
[`calculate_dispersion_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_dispersion_metrics.md),
[`calculate_lexical_dispersion()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_lexical_dispersion.md),
[`calculate_log_odds_ratio()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_log_odds_ratio.md),
[`calculate_text_readability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_text_readability.md),
[`clear_lexdiv_cache()`](https://mshin77.github.io/TextAnalysisR/reference/clear_lexdiv_cache.md),
[`detect_multi_words()`](https://mshin77.github.io/TextAnalysisR/reference/detect_multi_words.md),
[`extract_keywords_tfidf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_tfidf.md),
[`extract_morphology()`](https://mshin77.github.io/TextAnalysisR/reference/extract_morphology.md),
[`extract_named_entities()`](https://mshin77.github.io/TextAnalysisR/reference/extract_named_entities.md),
[`extract_noun_chunks()`](https://mshin77.github.io/TextAnalysisR/reference/extract_noun_chunks.md),
[`extract_pos_tags()`](https://mshin77.github.io/TextAnalysisR/reference/extract_pos_tags.md),
[`extract_subjects_objects()`](https://mshin77.github.io/TextAnalysisR/reference/extract_subjects_objects.md),
[`find_similar_words()`](https://mshin77.github.io/TextAnalysisR/reference/find_similar_words.md),
[`get_sentences()`](https://mshin77.github.io/TextAnalysisR/reference/get_sentences.md),
[`get_spacy_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/get_spacy_embeddings.md),
[`get_spacy_model_info()`](https://mshin77.github.io/TextAnalysisR/reference/get_spacy_model_info.md),
[`get_word_similarity()`](https://mshin77.github.io/TextAnalysisR/reference/get_word_similarity.md),
[`init_spacy_nlp()`](https://mshin77.github.io/TextAnalysisR/reference/init_spacy_nlp.md),
[`lexical_analysis`](https://mshin77.github.io/TextAnalysisR/reference/lexical_analysis.md),
[`lexical_diversity_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/lexical_diversity_analysis.md),
[`lexical_frequency_analysis()`](https://mshin77.github.io/TextAnalysisR/reference/lexical_frequency_analysis.md),
[`parse_morphology_string()`](https://mshin77.github.io/TextAnalysisR/reference/parse_morphology_string.md),
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
[`spacy_extract_entities()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_extract_entities.md),
[`spacy_has_vectors()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_has_vectors.md),
[`spacy_initialized()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_initialized.md),
[`spacy_lemmatize()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_lemmatize.md),
[`spacy_parse_full()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_parse_full.md),
[`summarize_morphology()`](https://mshin77.github.io/TextAnalysisR/reference/summarize_morphology.md)

## Examples

``` r
if (FALSE) { # \dontrun{
library(quanteda)
corp <- corpus(c("positive text", "negative text", "positive words"))
dfm_obj <- dfm(tokens(corp))
# Compare first document vs rest
keywords <- extract_keywords_keyness(dfm_obj, target = 1)
print(keywords)
} # }
```
