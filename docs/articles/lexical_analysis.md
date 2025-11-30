# Lexical Analysis

Lexical analysis examines word patterns and frequencies.

## Setup

``` r
library(TextAnalysisR)

mydata <- SpecialEduTech
united_tbl <- unite_cols(mydata, listed_vars = c("title", "keyword", "abstract"))
tokens <- prep_texts(united_tbl, text_field = "united_texts")
dfm_object <- quanteda::dfm(tokens)
```

## Word Frequency

``` r
plot_word_frequency(dfm_object, top_n = 20)
```

## Keyword Extraction

### TF-IDF

Find distinctive words per document:

``` r
keywords <- extract_keywords_tfidf(dfm_object, top_n = 10)
plot_tfidf_keywords(keywords, n_docs = 5)
```

### Keyness

Compare word usage between groups:

``` r
keyness <- extract_keywords_keyness(
  dfm_object,
  target_group = "Journal Article",
  reference_groups = "Conference Paper",
  category_var = "reference_type"
)
plot_keyness_keywords(keyness)
```

## Word Networks

### Co-occurrence

``` r
plot_cooccurrence_network(dfm_object, min_count = 10)
```

### Correlation

``` r
plot_correlation_network(dfm_object, min_cor = 0.3)
```
