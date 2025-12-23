# Preprocessing

Preprocessing cleans and prepares text for analysis.

## Workflow

``` r
library(TextAnalysisR)

# 1. Load data
mydata <- SpecialEduTech

# 2. Combine text columns
united_tbl <- unite_cols(mydata, listed_vars = c("title", "keyword", "abstract"))

# 3. Tokenize and clean
tokens <- prep_texts(
  united_tbl,
  text_field = "united_texts",
  remove_punct = TRUE,
  remove_numbers = TRUE
)

# 4. Remove stopwords
tokens_clean <- quanteda::tokens_remove(tokens, quanteda::stopwords("en"))

# 5. Create document-feature matrix
dfm_object <- quanteda::dfm(tokens_clean)
```

------------------------------------------------------------------------

**Unite Text Columns**

Unite combines multiple text columns into a single column for analysis.
Useful when text content is spread across multiple fields that should be
analyzed together.

**Examples:**

- **Survey Data:** Combine multiple open-ended response columns
- **Multi-field Text:** Merge title, abstract, and body fields
- **Comments:** Concatenate multiple comment or note columns

**Usage:** Select one or multiple text columns to combine. Columns are
concatenated with spaces between them. The united column becomes the
text source for all subsequent preprocessing and analysis steps.

**Learn More:** [tidyr Unite
Function](https://tidyr.tidyverse.org/reference/unite.html)

------------------------------------------------------------------------

**Tokenization Options**

Tokenization segments continuous text into individual units (tokens),
typically words, converting unstructured text into structured format for
computational analysis.

**Options:**

- **Lowercase:** Convert all text to lowercase to treat “Text” and
  “text” as identical
- **Remove Punctuation:** Strip punctuation marks like periods, commas,
  quotes
- **Remove Numbers:** Eliminate numeric digits (keep for technical
  texts)
- **Remove Symbols:** Remove special characters (@, \#, \$, etc.)
- **Remove URLs:** Identify and remove web addresses

| Parameter        | Default | Use Case                     |
|------------------|---------|------------------------------|
| `remove_punct`   | TRUE    | FALSE for sentiment analysis |
| `remove_numbers` | TRUE    | FALSE for quantitative text  |
| `lowercase`      | TRUE    | FALSE to preserve case       |

**Usage:** Select preprocessing options based on your analysis goals.
Sentence segmentation splits text into sentences before tokenization
when sentence structure is important (e.g., sentiment analysis).

**Learn More:** [quanteda Tokens
Documentation](https://quanteda.io/reference/tokens.html)

------------------------------------------------------------------------

**Stopword Removal**

Stopwords are common words (e.g., “the”, “is”, “and”) that appear
frequently but carry little meaningful content for analysis. Removing
them reduces noise and improves focus on content-bearing words.

**When to Remove:**

- **Topic Modeling:** Helps identify content themes by removing function
  words
- **Keyword Extraction:** Ensures meaningful terms rise to the top
- **Content Analysis:** Focuses on substantive vocabulary

**Usage:** Use predefined stopword lists (e.g., Snowball) or add custom
words. For sentiment analysis or syntactic studies, consider keeping
stopwords as they may carry important meaning.

``` r
tokens_clean <- quanteda::tokens_remove(tokens, quanteda::stopwords("en"))
```

**Learn More:** [stopwords Package
Documentation](https://search.r-project.org/CRAN/refmans/stopwords/html/stopwords.html)

------------------------------------------------------------------------

**Lemmatization**

Lemmatization reduces words to their base or dictionary form (lemma).
For example, “running”, “ran”, and “runs” all become “run”. This groups
related word forms together for more meaningful analysis.

**Comparison:**

- **Lemmatization:** Uses linguistic knowledge to produce valid
  dictionary words (studies → study)
- **Stemming:** Uses simple rules to chop word endings (studies → studi)
- **Advantage:** Lemmatization produces readable, meaningful base forms

**Usage:** Apply lemmatization after tokenization to consolidate word
variants. Particularly useful for topic modeling and keyword extraction
where grouping related forms improves interpretability. Requires Python
with spaCy.

**Learn More:** [spaCy Lemmatization
Guide](https://spacy.io/usage/linguistic-features#lemmatization)

------------------------------------------------------------------------

**Document-Feature Matrix (DFM)**

A Document-Feature Matrix (DFM) is a mathematical representation where
rows are documents, columns are unique tokens (features), and cells
contain frequency counts. It converts unstructured text into structured
numerical format for computational analysis.

**Process:**

- **Tokenization:** Text is split into individual tokens (words)
- **Vocabulary:** All unique tokens form the matrix columns
- **Counting:** Each document-token pair is counted
- **Sparse Matrix:** Efficient storage format for large corpora

**Usage:** The DFM is the foundation for all downstream analyses
including keyword extraction, topic modeling, and semantic analysis.
Create it after preprocessing (tokenization, stopword removal,
lemmatization).

``` r
dfm_object <- quanteda::dfm(tokens_clean)
```

**Learn More:** [quanteda DFM
Documentation](https://quanteda.io/reference/dfm.html)

------------------------------------------------------------------------

## Multi-word Expressions

Detect phrases like “machine learning”:

``` r
tokens <- detect_multi_words(tokens, min_count = 10)
```

## Next Steps

- [Lexical
  Analysis](https://mshin77.github.io/TextAnalysisR/articles/lexical_analysis.md)
- [Semantic
  Analysis](https://mshin77.github.io/TextAnalysisR/articles/semantic_analysis.md)
- [Topic
  Modeling](https://mshin77.github.io/TextAnalysisR/articles/topic_modeling.md)
