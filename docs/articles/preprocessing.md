# Preprocessing

Preprocessing cleans and prepares text for analysis.

## Workflow

``` r
library(TextAnalysisR)

# 1. Load data
mydata <- SpecialEduTech

# 2. Combine text columns
united_tbl <- unite_cols(mydata, listed_vars = c("title", "keyword", "abstract"))

# 3. Tokenize and clean
tokens <- prep_texts(
  united_tbl,
  text_field = "united_texts",
  remove_punct = TRUE,
  remove_numbers = TRUE
)

# 4. Remove stopwords
tokens_clean <- quanteda::tokens_remove(tokens, quanteda::stopwords("en"))

# 5. Create document-feature matrix
dfm_object <- quanteda::dfm(tokens_clean)
```

## Options

| Parameter        | Default | Use Case                     |
|------------------|---------|------------------------------|
| `remove_punct`   | TRUE    | FALSE for sentiment analysis |
| `remove_numbers` | TRUE    | FALSE for quantitative text  |
| `lowercase`      | TRUE    | FALSE to preserve case       |

## Multi-word Expressions

Detect phrases like “machine learning”:

``` r
tokens <- detect_multi_words(tokens, min_count = 10)
```

## Next Steps

- [Lexical
  Analysis](https://mshin77.github.io/TextAnalysisR/articles/lexical_analysis.md)
- [Semantic
  Analysis](https://mshin77.github.io/TextAnalysisR/articles/semantic_analysis.md)
- [Topic
  Modeling](https://mshin77.github.io/TextAnalysisR/articles/topic_modeling.md)
