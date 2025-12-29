# Wrap Text for Tooltip Display

Formats text for tooltip display with size limits and line wrapping.

## Usage

``` r
wrap_text_for_tooltip(
  text,
  max_words = 150,
  chars_per_line = 50,
  max_lines = 3
)
```

## Arguments

- text:

  Character string to format.

- max_words:

  Maximum words (not currently used, kept for compatibility).

- chars_per_line:

  Maximum characters per line (default: 50).

- max_lines:

  Maximum number of lines (default: 3).

## Value

Formatted text suitable for tooltip display.

## See also

Other text-utilities:
[`truncate_text_to_words()`](https://mshin77.github.io/TextAnalysisR/reference/truncate_text_to_words.md),
[`truncate_text_with_ellipsis()`](https://mshin77.github.io/TextAnalysisR/reference/truncate_text_with_ellipsis.md),
[`wrap_long_text()`](https://mshin77.github.io/TextAnalysisR/reference/wrap_long_text.md)
