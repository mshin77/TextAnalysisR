# Detect Corpus Language From Stopword Overlap

Scores texts against snowball stopword lists and ranks candidate
languages by the share of tokens matching each list. Uses only the
`stopwords` package, so no language-detection dependency is added.
Accuracy is good for European languages given a few hundred tokens, and
poor for very short texts or languages absent from snowball; treat the
result as a suggestion.

## Usage

``` r
detect_language(
  texts,
  languages = stopwords::stopwords_getlanguages("snowball"),
  sample_n = 200,
  seed = 123
)
```

## Arguments

- texts:

  Character vector of documents.

- languages:

  Candidate language codes (default: all snowball languages).

- sample_n:

  Maximum documents to sample (default 200).

- seed:

  Seed for sampling.

## Value

A tibble of `language` and `score` (share of tokens matching that
language's stopwords), ranked best first, or `NULL` when no tokens are
found. Ties and low scores mean the corpus language is unclear.

## See also

[`prep_texts()`](https://mshin77.github.io/TextAnalysisR/reference/prep_texts.md)
for the `stopwords_language` argument this informs.
