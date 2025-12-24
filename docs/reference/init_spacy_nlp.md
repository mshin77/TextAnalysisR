# Initialize spaCy NLP Module

Loads the Python spaCy module and initializes the NLP processor. This
function is called automatically by other spaCy functions.

## Usage

``` r
init_spacy_nlp(model = "en_core_web_sm", force = FALSE)
```

## Arguments

- model:

  Character; spaCy model to use. Options:

  - "en_core_web_sm" - Small model, fast, no word vectors (default)

  - "en_core_web_md" - Medium model, includes word vectors

  - "en_core_web_lg" - Large model, better accuracy, word vectors

  - "en_core_web_trf" - Transformer model, best accuracy

- force:

  Logical; force reinitialization even if already loaded

## Value

Invisible NULL. The spaCy processor is stored internally.

## Details

Requires Python with spaCy installed. Install with:

    pip install spacy
    python -m spacy download en_core_web_sm

## Examples

``` r
if (FALSE) { # \dontrun{
init_spacy_nlp("en_core_web_sm")
} # }
```
