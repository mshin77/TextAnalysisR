# Parse Texts with spaCy

Parse texts using spaCy and return token-level annotations. This is the
main parsing function for NLP analysis. Works with character vectors or
quanteda tokens objects.

## Usage

``` r
spacy_parse_full(
  x,
  pos = TRUE,
  tag = TRUE,
  lemma = TRUE,
  entity = FALSE,
  dependency = FALSE,
  morph = FALSE,
  model = "en_core_web_sm"
)
```

## Arguments

- x:

  Character vector of texts OR a quanteda tokens object.

- pos:

  Logical; include coarse POS tags (default: TRUE).

- tag:

  Logical; include fine-grained tags (default: TRUE).

- lemma:

  Logical; include lemmatized forms (default: TRUE).

- entity:

  Logical; include named entity tags (default: FALSE).

- dependency:

  Logical; include dependency relations (default: FALSE).

- morph:

  Logical; include morphological features (default: FALSE).

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with token-level annotations including:

- `doc_id`: Document identifier

- `sentence_id`: Sentence number within document

- `token_id`: Token position within sentence

- `token`: Original token text

- `pos`: Coarse POS tag (if pos = TRUE)

- `tag`: Fine-grained tag (if tag = TRUE)

- `lemma`: Lemmatized form (if lemma = TRUE)

- `entity`: Named entity tag (if entity = TRUE)

- `head_token_id`: Head token ID (if dependency = TRUE)

- `dep_rel`: Dependency relation (if dependency = TRUE)

- `morph`: Morphological features string (if morph = TRUE)

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
[`summarize_morphology()`](https://mshin77.github.io/TextAnalysisR/reference/summarize_morphology.md)

## Examples

``` r
if (FALSE) { # \dontrun{
# From SpecialEduTech dataset
texts <- TextAnalysisR::SpecialEduTech$abstract[1:5]
parsed <- spacy_parse_full(texts, morph = TRUE)

# From quanteda tokens
united <- unite_cols(TextAnalysisR::SpecialEduTech, c("title", "abstract"))
tokens <- prep_texts(united, text_field = "united_texts")
parsed <- spacy_parse_full(tokens, morph = TRUE)
} # }
```
