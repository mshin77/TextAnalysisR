# Lexical Diversity Analysis

Calculates multiple lexical diversity metrics for a document-feature
matrix (DFM) or tokens object. Supports all quanteda.textstats measures
plus MTLD (Measure of Textual Lexical Diversity), which is the most
recommended measure according to McCarthy & Jarvis (2010) for being
independent of text length.

## Usage

``` r
lexical_diversity_analysis(x, measures = "all", texts = NULL, cache_key = NULL)
```

## Arguments

- x:

  A quanteda DFM or tokens object. Tokens object is preferred for
  accurate MTLD calculation since it preserves token order.

- measures:

  Character vector of measures to calculate. Default is "all" which
  includes: TTR, C, R, CTTR, U, S, K, I, D, Vm, Maas, MATTR, MSTTR, and
  MTLD. Most recommended: "MTLD" or "MATTR" for length-independent
  measures.

- texts:

  Optional character vector of original texts. Required for MTLD
  calculation when using DFM input (since DFM loses token order).

- cache_key:

  Optional cache key (e.g., from digest::digest) for caching expensive
  calculations. Use the same cache_key to retrieve cached results.

## Value

A list containing:

- `lexical_diversity`: Data frame with per-document lexical diversity
  scores

- `summary_stats`: List of summary statistics (mean, median, sd) for
  each measure

## Details

MTLD (Measure of Textual Lexical Diversity) is calculated using the
algorithm from McCarthy & Jarvis (2010). It counts the number of
"factors" needed to reduce TTR below 0.72, then divides the number of
tokens by the number of factors. This provides a length-independent
measure of lexical diversity.

Important notes:

- For MTLD accuracy, pass a tokens object (not DFM) as input

- If using DFM, provide the 'texts' parameter for MTLD calculation

- MATTR and MSTTR window sizes are automatically adjusted for short
  documents

- Results are cached when cache_key is provided for repeated analysis

## References

McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A
validation study of sophisticated approaches to lexical diversity
assessment. Behavior Research Methods, 42(2), 381-392.

## See also

Other lexical:
[`calculate_dispersion_metrics()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_dispersion_metrics.md),
[`calculate_lexical_dispersion()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_lexical_dispersion.md),
[`calculate_log_odds_ratio()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_log_odds_ratio.md),
[`calculate_text_readability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_text_readability.md),
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
data(SpecialEduTech)
texts <- SpecialEduTech$abstract[1:10]
corp <- quanteda::corpus(texts)
toks <- quanteda::tokens(corp)
# Preferred: pass tokens object for accurate MTLD
lex_div <- lexical_diversity_analysis(toks, texts = texts)
# With caching for repeated analysis
cache_key <- digest::digest(texts)
lex_div <- lexical_diversity_analysis(toks, texts = texts, cache_key = cache_key)
# Alternative: pass DFM with texts for MTLD accuracy
dfm_obj <- quanteda::dfm(toks)
lex_div <- lexical_diversity_analysis(dfm_obj, texts = texts)
print(lex_div)
} # }
```
