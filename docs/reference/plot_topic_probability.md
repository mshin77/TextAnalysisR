# Plot Per-Document Per-Topic Probabilities

This function generates a bar plot showing the prevalence of each topic
across all documents.

## Usage

``` r
plot_topic_probability(
  stm_model = NULL,
  gamma_data = NULL,
  top_n = 10,
  height = 800,
  width = 1000,
  topic_labels = NULL,
  colors = NULL,
  verbose = TRUE,
  ...
)
```

## Arguments

- stm_model:

  A fitted STM model object. where `stm_model` is a fitted Structural
  Topic Model created using
  [`stm::stm()`](https://rdrr.io/pkg/stm/man/stm.html).

- gamma_data:

  Optional pre-computed gamma data frame (default: NULL). If provided,
  used instead of stm_model.

- top_n:

  The number of topics to display, ordered by their mean prevalence.

- height:

  The height of the resulting Plotly plot, in pixels (default: 800).

- width:

  The width of the resulting Plotly plot, in pixels (default: 1000).

- topic_labels:

  Optional topic labels (default: NULL).

- colors:

  Optional color palette for topics (default: NULL).

- verbose:

  Logical, if TRUE, prints progress messages.

- ...:

  Further arguments passed to
  [`tidytext::tidy`](https://generics.r-lib.org/reference/tidy.html).

## Value

A `ggplot` object showing a bar plot of topic prevalence. Topics are
ordered by their mean gamma value (average prevalence across documents).

## See also

Other visualization:
[`apply_standard_plotly_layout()`](https://mshin77.github.io/TextAnalysisR/reference/apply_standard_plotly_layout.md),
[`create_empty_plot_message()`](https://mshin77.github.io/TextAnalysisR/reference/create_empty_plot_message.md),
[`create_message_table()`](https://mshin77.github.io/TextAnalysisR/reference/create_message_table.md),
[`create_standard_ggplot_theme()`](https://mshin77.github.io/TextAnalysisR/reference/create_standard_ggplot_theme.md),
[`get_dt_options()`](https://mshin77.github.io/TextAnalysisR/reference/get_dt_options.md),
[`get_plotly_hover_config()`](https://mshin77.github.io/TextAnalysisR/reference/get_plotly_hover_config.md),
[`get_sentiment_color()`](https://mshin77.github.io/TextAnalysisR/reference/get_sentiment_color.md),
[`get_sentiment_colors()`](https://mshin77.github.io/TextAnalysisR/reference/get_sentiment_colors.md),
[`plot_cluster_terms()`](https://mshin77.github.io/TextAnalysisR/reference/plot_cluster_terms.md),
[`plot_cross_category_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_cross_category_heatmap.md),
[`plot_entity_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_entity_frequencies.md),
[`plot_mwe_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_mwe_frequency.md),
[`plot_ngram_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_ngram_frequency.md),
[`plot_pos_frequencies()`](https://mshin77.github.io/TextAnalysisR/reference/plot_pos_frequencies.md),
[`plot_semantic_viz()`](https://mshin77.github.io/TextAnalysisR/reference/plot_semantic_viz.md),
[`plot_similarity_heatmap()`](https://mshin77.github.io/TextAnalysisR/reference/plot_similarity_heatmap.md),
[`plot_term_trends_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_term_trends_continuous.md),
[`plot_topic_effects_categorical()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_categorical.md),
[`plot_topic_effects_continuous()`](https://mshin77.github.io/TextAnalysisR/reference/plot_topic_effects_continuous.md),
[`plot_word_frequency()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_frequency.md),
[`plot_word_probability()`](https://mshin77.github.io/TextAnalysisR/reference/plot_word_probability.md)

## Examples

``` r
if (interactive()) {

mydata <- TextAnalysisR::SpecialEduTech

 united_tbl <- TextAnalysisR::unite_cols(
   mydata,
   listed_vars = c("title", "keyword", "abstract")
 )

 tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")

 dfm_object <- quanteda::dfm(tokens)

 out <- quanteda::convert(dfm_object, to = "stm")

stm_15 <- stm::stm(
  data = out$meta,
  documents = out$documents,
  vocab = out$vocab,
  max.em.its = 75,
  init.type = "Spectral",
  K = 15,
  prevalence = ~ reference_type + s(year),
  verbose = TRUE)

topic_probability_plot <- TextAnalysisR::plot_topic_probability(
 stm_model = stm_15,
 top_n = 10,
 height = 800,
 width = 1000,
 verbose = TRUE)

print(topic_probability_plot)
}
```
