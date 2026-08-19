# Attach an Analytic Memo to a Unit or Category

Records the reasoning behind a coding decision so the interpretive step
leaves a trail. Memos attach to a single unit or to a category, carry
the round they were written in, and travel with the coded units on
export.

## Usage

``` r
add_memo(
  memos = NULL,
  target_type = c("unit", "category"),
  target_id,
  text,
  round = 1
)
```

## Arguments

- memos:

  Existing memo table, or `NULL` to start one.

- target_type:

  "unit" or "category".

- target_id:

  Identifier of the unit or category the memo describes.

- text:

  The memo. Free text; nothing parses it.

- round:

  Refinement round the memo belongs to (default 1).

## Value

The memo table with one row appended.

## See also

[`get_memos()`](https://mshin77.github.io/TextAnalysisR/reference/get_memos.md)
to retrieve them;
[`round_summary()`](https://mshin77.github.io/TextAnalysisR/reference/round_summary.md)
for the rounds memos are stamped with.
