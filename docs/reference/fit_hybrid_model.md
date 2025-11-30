# Fit Hybrid Topic Model

Fits a hybrid topic model combining STM with embedding-based methods.
This function integrates structural topic modeling (STM) with semantic
embeddings for enhanced topic discovery. The STM component provides
statistical rigor and covariate modeling capabilities, while the
embedding component adds semantic coherence.

**Effect Estimation:** Covariate effects on topic prevalence can be
estimated using the STM component via
[`stm::estimateEffect()`](https://rdrr.io/pkg/stm/man/estimateEffect.html).
The embedding component provides semantically meaningful topic
representations but does not support direct covariate modeling. This
hybrid approach combines the best of both worlds: statistical inference
from STM and semantic quality from embeddings.

## Usage

``` r
fit_hybrid_model(
  texts,
  metadata = NULL,
  n_topics_stm = 10,
  embedding_model = "all-MiniLM-L6-v2",
  stm_prevalence = NULL,
  stm_init_type = "Spectral",
  alignment_method = "cosine",
  verbose = TRUE,
  seed = 123
)
```

## Arguments

- texts:

  A character vector of texts to analyze.

- metadata:

  Optional data frame with document metadata for STM covariate modeling.

- n_topics_stm:

  Number of topics for STM (default: 10).

- embedding_model:

  Embedding model name (default: "all-MiniLM-L6-v2").

- stm_prevalence:

  Formula for STM prevalence covariates (e.g., ~ category + s(year,
  df=3)).

- stm_init_type:

  STM initialization type (default: "Spectral").

- alignment_method:

  Method for aligning STM and embedding topics (default: "cosine").

- verbose:

  Logical, if TRUE, prints progress messages.

- seed:

  Random seed for reproducibility.

## Value

A list containing:

- stm_result: The STM model output (use this for effect estimation)

- embedding_result: The embedding-based topic model output

- alignment: Alignment metrics between the two models

- combined_topics: Integrated topic representations

- metadata: Metadata used in modeling (needed for effect estimation)

## Note

For covariate effect estimation, use
[`stm::estimateEffect()`](https://rdrr.io/pkg/stm/man/estimateEffect.html)
on the `stm_result$model` component. The metadata must include the
covariates specified in `stm_prevalence`.

## Examples

``` r
if (FALSE) { # \dontrun{
  texts <- c("Computer-assisted instruction improves math skills for students with disabilities",
             "Assistive technology supports reading comprehension for learning disabled students",
             "Mobile devices enhance communication for students with autism spectrum disorder")

  hybrid_model <- fit_hybrid_model(
    texts = texts,
    n_topics_stm = 3,
    verbose = TRUE
  )
} # }
```
