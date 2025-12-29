# Extract Part-of-Speech Tags from Tokens

Uses spaCy to extract part-of-speech (POS) tags from tokenized text.
Returns a data frame with token-level POS annotations.

## Usage

``` r
extract_pos_tags(
  tokens,
  include_lemma = TRUE,
  include_entity = FALSE,
  include_dependency = FALSE,
  model = "en_core_web_sm"
)
```

## Arguments

- tokens:

  A quanteda tokens object or character vector of texts.

- include_lemma:

  Logical; include lemmatized forms (default: TRUE).

- include_entity:

  Logical; include named entity recognition (default: FALSE).

- include_dependency:

  Logical; include dependency parsing (default: FALSE).

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with columns:

- `doc_id`: Document identifier

- `sentence_id`: Sentence number within document

- `token_id`: Token position within sentence

- `token`: Original token

- `pos`: Universal POS tag (e.g., NOUN, VERB, ADJ)

- `tag`: Detailed POS tag (e.g., NN, VBD, JJ)

- `lemma`: Lemmatized form (if include_lemma = TRUE)

- `entity`: Named entity type (if include_entity = TRUE)

- `head_token_id`: Head token in dependency tree (if include_dependency
  = TRUE)

- `dep_rel`: Dependency relation type, e.g., nsubj, dobj (if
  include_dependency = TRUE)

## Details

This function requires the Python with spaCy installed. If spaCy is not
initialized, this function will attempt to initialize it with the
specified model.

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
[`spacy_parse_full()`](https://mshin77.github.io/TextAnalysisR/reference/spacy_parse_full.md),
[`summarize_morphology()`](https://mshin77.github.io/TextAnalysisR/reference/summarize_morphology.md)

## Examples

``` r
if (FALSE) { # \dontrun{
tokens <- quanteda::tokens("The quick brown fox jumps over the lazy dog.")
pos_data <- extract_pos_tags(tokens)
print(pos_data)
} # }
```
