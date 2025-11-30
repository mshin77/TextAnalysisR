# Analyze Text Sentiment

Performs sentiment analysis on text data using the syuzhet package.
Returns sentiment scores and classifications.

## Usage

``` r
analyze_sentiment(texts, method = "syuzhet", doc_ids = NULL)
```

## Arguments

- texts:

  Character vector of texts to analyze

- method:

  Sentiment analysis method: "syuzhet", "bing", "afinn", or "nrc"
  (default: "syuzhet")

- doc_ids:

  Optional character vector of document identifiers (default: NULL)

## Value

A data frame with columns:

- document:

  Document identifier

- text:

  Original text

- sentiment_score:

  Numeric sentiment score

- sentiment:

  Classification: "positive", "negative", or "neutral"

## Examples

``` r
if (FALSE) { # \dontrun{
texts <- c(
  "This research shows promising results for students.",
  "The intervention had no significant effect.",
  "Students struggled with the complex material."
)
results <- analyze_sentiment(texts)
print(results)
} # }
```
