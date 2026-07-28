![TextAnalysisR Logo](reference/figures/logo.png)

[![R-CMD-check](https://github.com/mshin77/TextAnalysisR/workflows/R-CMD-check/badge.svg)](https://github.com/mshin77/TextAnalysisR/actions)
[![CRAN
status](https://www.r-pkg.org/badges/version/TextAnalysisR)](https://CRAN.R-project.org/package=TextAnalysisR)
[![Project Status:
Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License:
GPL-3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

Text mining and natural language processing workflow for documents
(`PDF`, `DOCX`, `XLSX`, `CSV`, `TXT`). Includes preprocessing via
[quanteda](https://github.com/quanteda/quanteda), lexical analysis (term
frequency-inverse document frequency, log-odds ratios, lexical
diversity) via [tidytext](https://github.com/juliasilge/tidytext), topic
modeling via [stm](https://github.com/bstewart/stm) and
[BERTopic](https://maartengr.github.io/BERTopic/), semantic similarity
and document clustering on transformer embeddings, an interactive
[Shiny](https://shiny.posit.co/) interface with
[ggplot2](https://ggplot2.tidyverse.org/) visualization, optional
[spaCy](https://spacy.io/) lemmatization, and local
[sentence-transformers](https://www.sbert.net/) or web-based (OpenAI,
[Gemini](https://ai.google.dev/)) model providers for
retrieval-augmented generation.

## Installation

Release version from CRAN:

``` R
install.packages("TextAnalysisR")
```

Development version from R-universe:

``` R
install.packages("TextAnalysisR", repos = "https://mshin77.r-universe.dev")
```

## Python Setup (Optional)

Core analyses run in plain R. Python is only needed for lemmatization,
embeddings, clustering, PDF extraction, and transformer-based analyses.
Run once after installing:

``` R
library(TextAnalysisR)
setup_python_env()
```

This sets up a dedicated virtualenv with the required Python packages.
Restart R afterward; check status with
[`check_python_env()`](https://mshin77.github.io/TextAnalysisR/reference/check_python_env.md).

## Load the TextAnalysisR Package

``` R
library(TextAnalysisR)
```

## Alternatively, Launch and Browse the Shiny App

Access the web app at <https://www.textanalysisr.org>.

Launch and browse the app on the local computer:

``` R
run_app()
```

## Getting Started

See [Quick
Start](https://mshin77.github.io/TextAnalysisR//articles/quickstart.html)
for tutorials.

## Citation

- Shin, M. (2026). *TextAnalysisR: A text mining workflow tool* (R
  package version 0.1.4) \[Computer software\].
  <https://mshin77.github.io/TextAnalysisR/>

- Shin, M. (2026). *TextAnalysisR: A text mining workflow tool* \[Web
  application\]. <https://www.textanalysisr.org>
