# Lexical Analysis

Lexical analysis examines word patterns and frequencies.

## Setup

``` r
library(TextAnalysisR)

mydata <- SpecialEduTech
united_tbl <- unite_cols(mydata, listed_vars = c("title", "keyword", "abstract"))
tokens <- prep_texts(united_tbl, text_field = "united_texts")
dfm_object <- quanteda::dfm(tokens)
```

## Word Frequency

``` r
plot_word_frequency(dfm_object, top_n = 20)
```

------------------------------------------------------------------------

**TF-IDF Keyword Extraction**

Find distinctive words per document using Term Frequency-Inverse
Document Frequency:

``` r
keywords <- extract_keywords_tfidf(dfm_object, top_n = 10)
plot_tfidf_keywords(keywords, n_docs = 5)
```

TF-IDF weights terms that are frequent in a document but rare across the
corpus, identifying distinctive vocabulary.

------------------------------------------------------------------------

**Keyness Analysis**

Compare word usage between groups:

``` r
keyness <- extract_keywords_keyness(
  dfm_object,
  target_group = "Journal Article",
  reference_groups = "Conference Paper",
  category_var = "reference_type"
)
plot_keyness_keywords(keyness)
```

Keyness analysis identifies statistically significant differences in
word usage between groups.

------------------------------------------------------------------------

**N-gram Analysis**

N-grams are sequences of consecutive words that frequently appear
together. They capture multi-word expressions like “machine learning” or
“New York City” that carry meaning as complete phrases.

**Types:**

- **Bigrams:** 2-word sequences (e.g., “data analysis”)
- **Trigrams:** 3-word sequences (e.g., “natural language processing”)
- **4-grams & 5-grams:** Longer phrases (e.g., “statistical significance
  test results”)

**Usage:** Set minimum frequency (how often phrases appear) and lambda
(collocation strength) to detect meaningful multi-word expressions.

``` r
tokens <- detect_multi_words(tokens, min_count = 10)
```

**Learn More:** [Text Mining with R - N-grams
Chapter](https://www.tidytextmining.com/ngrams.html)

------------------------------------------------------------------------

**Part-of-Speech Tagging**

Part-of-speech (POS) tagging identifies the grammatical category of each
word. Requires Python with spaCy.

**Tags (Universal Dependencies):**

- **NOUN, VERB, ADJ, ADV:** Content words
- **PROPN:** Proper nouns (names)
- **DET, ADP, PRON:** Function words
- **NUM, PUNCT:** Numbers, punctuation

**Usage:** Filter by tags to focus on specific word types (e.g., nouns
and verbs for content analysis).

**Learn More:** [Universal Dependencies POS
Tags](https://universaldependencies.org/u/pos/)

------------------------------------------------------------------------

**Morphological Analysis**

Morphological analysis extracts grammatical features from words. Uses
Python spaCy via reticulate.

**Features:**

| Feature  | Description        | Values              |
|----------|--------------------|---------------------|
| Number   | Singular/Plural    | Sing, Plur          |
| Tense    | Verb tense         | Past, Pres, Fut     |
| VerbForm | Verb form          | Fin, Inf, Part, Ger |
| Person   | Grammatical person | 1, 2, 3             |
| Case     | Grammatical case   | Nom, Acc, Dat, Gen  |

``` r
parsed <- extract_pos_tags(texts)  # Uses spacy_parse_full
# Returns columns: doc_id, token, lemma, pos, tag
```

**Usage:** Analyze verb tenses for temporal patterns, number agreement,
or grammatical complexity.

**Learn More:** [spaCy
Morphology](https://spacy.io/usage/linguistic-features#morphology)

------------------------------------------------------------------------

**Named Entity Recognition**

Named Entity Recognition (NER) identifies and classifies named entities
in text. Requires Python with spaCy.

**Entity Types:**

- **PERSON, ORG:** People, organizations
- **GPE, LOC:** Places, locations
- **DATE, MONEY, PERCENT:** Temporal, monetary values

**Usage:** Filter by entity type. Add custom entities for qualitative
coding.

**Learn More:** [spaCy Named Entity
Recognition](https://spacy.io/usage/linguistic-features#named-entities)

------------------------------------------------------------------------

## Word Networks

### Co-occurrence

``` r
word_co_occurrence_network(dfm_object, co_occur_n = 10)
```

### Correlation

``` r
word_correlation_network(dfm_object, corr_n = 0.3)
```

------------------------------------------------------------------------

**Log Odds Ratio Analysis**

Log odds ratio compares word frequencies between categories to identify
distinctive vocabulary.

**Simple Log Odds Ratio:**

``` r
log_odds <- calculate_log_odds_ratio(
  dfm_object,
  group_var = "category",
  comparison_mode = "binary",
  top_n = 15
)
plot_log_odds_ratio(log_odds)
```

**Weighted Log Odds Ratio:**

For publication-quality analysis, use the weighted log odds method which
accounts for sampling variability by weighting results with z-scores.
This method identifies words that reliably distinguish between groups,
not just rare words with extreme ratios.

``` r
# Requires tidylo package: install.packages("tidylo")
weighted_odds <- calculate_weighted_log_odds(
  dfm_object,
  group_var = "category",
  top_n = 15
)
```

**Learn More:** [tidylo: Weighted Log
Odds](https://juliasilge.github.io/tidylo/)

------------------------------------------------------------------------

**Lexical Dispersion**

Lexical dispersion (X-ray plot) shows where terms appear across
documents.

``` r
dispersion <- calculate_lexical_dispersion(tokens, terms = c("education", "technology"))
plot_lexical_dispersion(dispersion)
```

------------------------------------------------------------------------

**Readability Metrics**

Readability metrics quantify text complexity using statistical measures
of sentence structure and word characteristics.

**Available Metrics:**

| Metric | Formula Basis | Output |
|----|----|----|
| Flesch Reading Ease | Sentence length + syllables | 0-100 (higher = easier) |
| Flesch-Kincaid | Sentence length + syllables | U.S. grade level |
| Gunning Fog | Sentence length + complex words | Years of education |
| SMOG | Polysyllabic word count | Years of education |
| ARI | Characters per word | U.S. grade level |
| Coleman-Liau | Letters per 100 words | U.S. grade level |

**Usage Notes:**

- Different formulas may produce slightly different grade level
  estimates
- These formulas measure surface-level text features (word length,
  sentence length)
- Short texts may produce less reliable scores

``` r
readability <- calculate_text_readability(united_tbl$united_texts)
plot_readability_distribution(readability)
```

**Learn More:** [quanteda textstat_readability
Documentation](https://quanteda.io/reference/textstat_readability.html)

------------------------------------------------------------------------

**Lexical Diversity Metrics**

Lexical diversity measures vocabulary richness by quantifying the
relationship between unique words (types) and total words (tokens).

**Available Metrics:**

| Metric | Description                 | Note                          |
|--------|-----------------------------|-------------------------------|
| TTR    | Types / Tokens              | Sensitive to text length      |
| CTTR   | Types / sqrt(2 × Tokens)    | Partially corrects for length |
| MSTTR  | Mean Segmental TTR          | Divides into segments         |
| MATTR  | Moving Average TTR          | More stable across lengths    |
| MTLD   | Mean length maintaining TTR | Text-length independent       |
| Maas   | Log-based formula           | Lower = more diverse          |

**Usage Notes:**

- MTLD and MATTR are more stable across different text lengths
- TTR is sensitive to text length - compare only similar-length texts
- Maas, Yule K, and Simpson D use inverse scales (lower = more diverse)

``` r
diversity <- lexical_diversity_analysis(dfm_object)
plot_lexical_diversity_distribution(diversity)
```

**Learn More:** [quanteda textstat_lexdiv
Documentation](https://quanteda.io/reference/textstat_lexdiv.html)

------------------------------------------------------------------------

## Next Steps

- [Semantic
  Analysis](https://mshin77.github.io/TextAnalysisR/articles/semantic_analysis.md)
- [Topic
  Modeling](https://mshin77.github.io/TextAnalysisR/articles/topic_modeling.md)
