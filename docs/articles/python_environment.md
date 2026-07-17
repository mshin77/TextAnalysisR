# Python Environment

``` r

library(TextAnalysisR)

tokens <- quanteda::tokens(SpecialEduTech$abstract[1:5])
dispersion <- calculate_lexical_dispersion(
  tokens,
  terms = c("learning", "instruction")
)
head(dispersion)
```

    ##   doc_id        term  position doc_length
    ## 1  text2    learning 0.5200000         25
    ## 2  text3    learning 0.3714286         35
    ## 3  text3 instruction 0.3142857         35
    ## 4  text3 instruction 0.6857143         35
    ## 5  text5    learning 0.7904762        105

Python enables features: NLP with spaCy, embeddings, and neural
sentiment analysis.

## Quick Setup

[`setup_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/setup_python_env.md)
automatically:

1.  Creates virtual environment `textanalysisr-env`
2.  Installs `spacy` and `pdfplumber`
3.  Downloads spaCy English model (`en_core_web_sm`)

Uses virtualenv (or conda if available).

## Check Status

Run
[`check_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/check_python_env.md)
to verify the environment.

## Common Issues

### “Another Python already initialized”

Set preferred environment in `.Rprofile` with
`Sys.setenv(RETICULATE_PYTHON_ENV = "textanalysisr-env")`, then restart
R.

### Environment in OneDrive

Avoid OneDrive paths. Use
`setup_python_env(envname = "textanalysisr-env")`.

## spaCy Models

The default `en_core_web_sm` model is installed automatically. For word
vectors (similarity), the medium model is 91 MB and the large model is
560 MB:

``` bash
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

## Deep Learning (Optional)

For embeddings and neural sentiment:

``` bash
pip install sentence-transformers transformers torch
```

## Diagnostics

Use
[`reticulate::py_config()`](https://rstudio.github.io/reticulate/reference/py_config.html)
and
[`reticulate::virtualenv_list()`](https://rstudio.github.io/reticulate/reference/virtualenv-tools.html)
to inspect the active Python.
