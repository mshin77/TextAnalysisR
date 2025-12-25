# Extract Morphological Features

Uses spaCy to extract comprehensive morphological features from text.
Returns data with Number, Tense, VerbForm, Person, Case, Mood, Aspect,
etc.

## Usage

``` r
extract_morphology(
  tokens,
  features = c("Number", "Tense", "VerbForm", "Person", "Case", "Mood", "Aspect"),
  include_pos = TRUE,
  include_lemma = TRUE,
  model = "en_core_web_sm"
)
```

## Arguments

- tokens:

  A quanteda tokens object or character vector of texts.

- features:

  Character vector of morphological features to extract. Default
  includes common Universal Dependencies features.

- include_pos:

  Logical; include POS tags (default: TRUE).

- include_lemma:

  Logical; include lemmatized forms (default: TRUE).

- model:

  Character; spaCy model to use (default: "en_core_web_sm").

## Value

A data frame with token-level morphological annotations including
morph\_\* columns for each requested feature.

## Details

Morphological features follow Universal Dependencies annotation. Common
features include:

- `Number`: Sing (singular), Plur (plural)

- `Tense`: Past, Pres (present), Fut (future)

- `VerbForm`: Fin (finite), Inf (infinitive), Part (participle), Ger
  (gerund)

- `Person`: 1, 2, 3 (first, second, third person)

- `Case`: Nom (nominative), Acc (accusative), Gen (genitive), Dat
  (dative)

- `Mood`: Ind (indicative), Imp (imperative), Sub (subjunctive)

- `Aspect`: Perf (perfective), Imp (imperfective), Prog (progressive)

## See also

Other lexical:
[`calculate_text_readability()`](https://mshin77.github.io/TextAnalysisR/reference/calculate_text_readability.md),
[`clear_lexdiv_cache()`](https://mshin77.github.io/TextAnalysisR/reference/clear_lexdiv_cache.md),
[`detect_multi_words()`](https://mshin77.github.io/TextAnalysisR/reference/detect_multi_words.md),
[`extract_keywords_keyness()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_keyness.md),
[`extract_keywords_tfidf()`](https://mshin77.github.io/TextAnalysisR/reference/extract_keywords_tfidf.md),
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
tokens <- quanteda::tokens("The cats are running quickly.")
morph_data <- extract_morphology(tokens)
print(morph_data)
} # }
```
