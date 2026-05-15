![TextAnalysisR Logo](reference/figures/logo.png)

[![R-CMD-check](https://github.com/mshin77/TextAnalysisR/workflows/R-CMD-check/badge.svg)](https://github.com/mshin77/TextAnalysisR/actions)
[![Project Status:
Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License:
GPL-3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

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
([Ollama](https://ollama.com),
[sentence-transformers](https://www.sbert.net/)) or web-based
([OpenAI](https://platform.openai.com/),
[Gemini](https://ai.google.dev/)) model providers for
retrieval-augmented generation.

## Installation

From [R-universe](https://mshin77.r-universe.dev) (pre-built binaries
for Windows, macOS, and Linux):

``` R
install.packages("TextAnalysisR",
  repos = c("https://mshin77.r-universe.dev", "https://cloud.r-project.org"))
```

Or the development version from
[GitHub](https://github.com/mshin77/TextAnalysisR):

``` R
install.packages("remotes")
remotes::install_github("mshin77/TextAnalysisR")
```

## First-Time Python Setup

Several functions
([`lemmatize_tokens()`](https://mshin77.github.io/TextAnalysisR/reference/lemmatize_tokens.md),
[`generate_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/generate_embeddings.md),
[`cluster_embeddings()`](https://mshin77.github.io/TextAnalysisR/reference/cluster_embeddings.md),
PDF extraction, transformer-based analyses) require Python packages. Run
this **once** after installing TextAnalysisR:

``` R
library(TextAnalysisR)
setup_python_env()
```

This creates a dedicated virtualenv (`textanalysisr-env`), installs the
packages listed in `inst/python/requirements.txt` (spaCy, pandas,
pdfplumber, sentence-transformers, torch, umap-learn, hdbscan,
scikit-learn, numba), and downloads the `en_core_web_sm` spaCy model.
Restart R afterward. Check status anytime with
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
