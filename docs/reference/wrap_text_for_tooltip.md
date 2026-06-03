# Wrap text for tooltip display

Wrap text for tooltip display

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

  Input string.

- max_words:

  Maximum words to keep.

- chars_per_line:

  Approximate line width.

- max_lines:

  Maximum lines.

## Value

A string with HTML `<br>` breaks.
