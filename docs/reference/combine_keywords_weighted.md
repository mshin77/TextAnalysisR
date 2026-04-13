# Combine Topic Keywords with Semantic Weighting

Combines keywords from STM and embedding-based topics using weighted
term co-associations and semantic similarity. Based on ensemble topic
modeling research (Belford et al., 2018).

## Usage

``` r
combine_keywords_weighted(
  stm_words,
  stm_probs = NULL,
  embed_words,
  embed_ranks = NULL,
  n_keywords = 10,
  stm_weight = 0.5
)
```

## Arguments

- stm_words:

  Character vector of STM topic words.

- stm_probs:

  Numeric vector of word probabilities from STM.

- embed_words:

  Character vector of embedding topic words.

- embed_ranks:

  Numeric vector of word ranks (1 = top word).

- n_keywords:

  Number of keywords to return (default: 10).

- stm_weight:

  Weight for STM words (default: 0.5).

## Value

A list containing:

- combined_words: Combined keyword list

- word_scores: Score for each word

- source: Source of each word (stm, embedding, or both)
