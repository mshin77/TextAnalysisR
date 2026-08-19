# Split Texts Into Units of Analysis

Cuts each document into the unit every stage of an analysis should
share. Character offsets are returned alongside each unit, so any span
stays recoverable from the original text.

## Usage

``` r
split_texts(texts, unit = c("sentence", "paragraph", "document"))
```

## Arguments

- texts:

  Character vector of documents, optionally named. Names become
  `doc_id`; positions are used when absent.

- unit:

  "sentence" (default), "paragraph", or "document". Paragraphs split on
  blank lines; sentences split after `.`, `!`, or `?`.

## Value

A tibble with `doc_id`, `unit_id`, `unit_text`, `start`, and `end`.
`unit_id` is the `doc_id` itself for whole documents, and `doc_id.n`
otherwise. Empty and missing documents contribute no rows.

## See also

[`apply_codes()`](https://mshin77.github.io/TextAnalysisR/reference/apply_codes.md),
which splits with this before coding;
[`uncoded_units()`](https://mshin77.github.io/TextAnalysisR/reference/uncoded_units.md)
for the units a codebook did not reach.
