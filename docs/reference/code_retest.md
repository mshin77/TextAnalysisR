# AI Coding Retest Stability

Codes the same sample of texts more than once at identical settings and
reports how often the runs agree, next to a shuffled-label baseline. Low
temperature does not guarantee stable output, so retest agreement is
measured rather than assumed. The baseline shows the agreement expected
if codes were unrelated to the texts; retest agreement should sit well
above it.

## Usage

``` r
code_retest(texts, codebook, n_runs = 2, sample_n = 50, seed = 123, ...)
```

## Arguments

- texts:

  Character vector of documents. Names become `doc_id`.

- codebook:

  Data frame with `code` and `definition` columns.

- n_runs:

  Number of coding runs (default 2).

- sample_n:

  Maximum number of documents to sample (default 50).

- seed:

  Seed for document sampling and the shuffled baseline.

- ...:

  Passed to
  [`apply_codes()`](https://mshin77.github.io/TextAnalysisR/reference/apply_codes.md)
  (`unit`, `max_codes`, `provider`, `api_key`, `delay`, ...).

## Value

A list with `summary` (tibble of metric, estimate, n_units, n_runs) and
`runs` (all assignments, with a `run` column), or `invisible(NULL)` when
fewer than two runs complete.

## See also

[`apply_codes()`](https://mshin77.github.io/TextAnalysisR/reference/apply_codes.md);
[`code_agreement()`](https://mshin77.github.io/TextAnalysisR/reference/code_agreement.md)
for human inter-coder reliability, which this does not replace.
