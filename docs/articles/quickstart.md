# Getting Started

TextAnalysisR provides text analysis through an interactive Shiny app or
R code.

## Install

``` r
install.packages("remotes")
remotes::install_github("mshin77/TextAnalysisR")
```

## Launch App

``` r
library(TextAnalysisR)
run_app()
```

Or visit [textanalysisr.org](https://www.textanalysisr.org) for the web
version.

## Quick Example

``` r
library(TextAnalysisR)

# Load data
mydata <- SpecialEduTech

# Combine text columns
united_tbl <- unite_cols(mydata, listed_vars = c("title", "keyword", "abstract"))

# Preprocess
tokens <- prep_texts(united_tbl, text_field = "united_texts")
dfm_object <- quanteda::dfm(tokens)

# Visualize
plot_word_frequency(dfm_object, top_n = 20)
```

## Features

| Category       | Analyses                           |
|----------------|------------------------------------|
| Lexical        | Word frequency, keywords, networks |
| Semantic       | Similarity, clustering, sentiment  |
| Topic Modeling | STM, BERTopic, hybrid              |

## Next Steps

- [Installation](https://mshin77.github.io/TextAnalysisR/articles/installation.md) -
  Full setup guide
- [Preprocessing](https://mshin77.github.io/TextAnalysisR/articles/preprocessing.md) -
  Prepare text data
- [Lexical
  Analysis](https://mshin77.github.io/TextAnalysisR/articles/lexical_analysis.md) -
  Word patterns
- [Semantic
  Analysis](https://mshin77.github.io/TextAnalysisR/articles/semantic_analysis.md) -
  Meaning and similarity
- [Topic
  Modeling](https://mshin77.github.io/TextAnalysisR/articles/topic_modeling.md) -
  Discover themes
