# Units With No Code Assigned

Returns the units
[`apply_codes()`](https://mshin77.github.io/TextAnalysisR/reference/apply_codes.md)
left uncoded, paired with their text. These units are the ones a
codebook does not reach. Reviewing them shows whether content recurs in
the corpus that the coding frame omits, and is the step that keeps a
codebook-constrained analysis from reporting only what the codebook was
built to find.

## Usage

``` r
uncoded_units(assignments, texts)
```

## Arguments

- assignments:

  Tibble from
  [`apply_codes()`](https://mshin77.github.io/TextAnalysisR/reference/apply_codes.md),
  with `doc_id`, `unit_id`, `start`, `end`, and `code`.

- texts:

  Character vector of the documents passed to
  [`apply_codes()`](https://mshin77.github.io/TextAnalysisR/reference/apply_codes.md).
  Names become `doc_id`; unnamed vectors are keyed by position.

## Value

A tibble of `doc_id`, `unit_id`, `start`, `end`, and `unit_text`, one
row per uncoded unit, ordered as in `assignments`. Zero rows when every
unit received at least one code.

## See also

[`apply_codes()`](https://mshin77.github.io/TextAnalysisR/reference/apply_codes.md)
to produce assignments.
