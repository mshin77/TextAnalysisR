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
[`calculate_dispersion_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_dispersion_metrics.md),
[`calculate_lexical_dispersion()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_lexical_dispersion.md),
[`calculate_log_odds_ratio()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_log_odds_ratio.md),
[`clear_lexdiv_cache()`](https://mshin77.github.io/TextAnalysisR/reference/clear_lexdiv_cache.md),
[`detect_multi_words()`](https://mshin77.github.io/TextAnalysisR/reference/detect_multi_words.md),
[`extract_keywords_keyness()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_keyness.md),
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
# \donttest{
data(SpecialEduTech, package = "TextAnalysisR")
texts <- SpecialEduTech$abstract[1:10]
readability <- calculate_text_readability(texts)
print(readability)
#>    Document     flesch flesch_kincaid gunning_fog Lexical Diversity (TTR)
#> 1     Doc 1  19.824902       17.53294    21.21569               0.7831325
#> 2     Doc 2  -5.024231       20.41923    24.24615               0.9166667
#> 3     Doc 3   5.505682       16.59045    22.35758               0.7812500
#> 4     Doc 4  27.617151       14.74849    17.90233               0.9069767
#> 5     Doc 5  16.490000       17.17000    20.00000               0.5757576
#> 6     Doc 6  14.513863       16.77798    21.36197               0.4415205
#> 7     Doc 7 -10.401667       19.18185    24.23704               0.8823529
#> 8     Doc 8  26.795461       15.57450    19.74545               0.4392157
#> 9     Doc 9  15.338100       16.87522    18.47530               0.5882353
#> 10   Doc 10  -2.587500       19.58250    22.93333               0.8510638
#>    Avg Sentence Length
#> 1             27.33333
#> 2             24.00000
#> 3             16.00000
#> 4             21.50000
#> 5             24.75000
#> 6             21.68750
#> 7             17.00000
#> 8             18.85714
#> 9             22.50000
#> 10            23.50000
# }
```
