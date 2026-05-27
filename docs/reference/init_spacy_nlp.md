# Initialize spaCy NLP

Initialize the spaCy NLP pipeline with the specified model. Uses a
cached instance for efficiency.

## Usage

``` r
init_spacy_nlp(model = "en_core_web_sm", force = FALSE)
```

## Arguments

- model:

  Character; spaCy model name (default: "en_core_web_sm").

- force:

  Logical; force reinitialization even if already initialized.

## Value

Invisibly returns the SpacyNLP Python object.

## Details

Available models:

- `en_core_web_sm`: Small English model (fast, no word vectors)

- `en_core_web_md`: Medium English model (word vectors)

- `en_core_web_lg`: Large English model (best accuracy)

## Examples

``` r
if (interactive()) {
init_spacy_nlp("en_core_web_sm")
}
```
