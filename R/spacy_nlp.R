#' @title spaCy NLP Functions via Python
#'
#' @description
#' Comprehensive spaCy integration using Python via reticulate.
#' Provides access to all spaCy linguistic features including morphology,
#' dependency parsing, named entity recognition, and word vectors.
#'
#' @name spacy_nlp
#' @family nlp
NULL

# Module-level cache for the Python spaCy module
.spacy_env <- new.env(hash = TRUE, parent = emptyenv())

#' Initialize spaCy NLP Module
#'
#' @description
#' Loads the Python spaCy module and initializes the NLP processor.
#' This function is called automatically by other spaCy functions.
#'
#' @param model Character; spaCy model to use. Options:
#'   \itemize{
#'     \item "en_core_web_sm" - Small model, fast, no word vectors (default)
#'     \item "en_core_web_md" - Medium model, includes word vectors
#'     \item "en_core_web_lg" - Large model, better accuracy, word vectors
#'     \item "en_core_web_trf" - Transformer model, best accuracy
#'   }
#' @param force Logical; force reinitialization even if already loaded
#'
#' @return Invisible NULL. The spaCy processor is stored internally.
#'
#' @details
#' Requires Python with spaCy installed. Install with:
#' \preformatted{
#' pip install spacy
#' python -m spacy download en_core_web_sm
#' }
#'
#' @export
#'
#' @examples
#' \dontrun{
#' init_spacy_nlp("en_core_web_sm")
#' }
init_spacy_nlp <- function(model = "en_core_web_sm", force = FALSE) {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required. Install with: install.packages('reticulate')")
  }

  # Check if already initialized with same model
  if (!force && exists("nlp", envir = .spacy_env) &&
      exists("model", envir = .spacy_env) &&
      get("model", envir = .spacy_env) == model) {
    return(invisible(NULL))
  }

  # Import the Python module
  python_path <- system.file("python", package = "TextAnalysisR")

  if (!dir.exists(python_path)) {
    stop("Python module directory not found. Package may not be installed correctly.")
  }

  tryCatch({
    spacy_module <- reticulate::import_from_path("spacy_nlp", path = python_path)
    nlp <- spacy_module$SpacyNLP(model)

    assign("module", spacy_module, envir = .spacy_env)
    assign("nlp", nlp, envir = .spacy_env)
    assign("model", model, envir = .spacy_env)

    message("spaCy initialized with model: ", model)
  }, error = function(e) {
    stop(
      "Failed to initialize spaCy. Ensure Python and spaCy are installed:\n",
      "  pip install spacy\n",
      "  python -m spacy download ", model, "\n",
      "Error: ", e$message

    )
  })

  invisible(NULL)
}


#' Get spaCy NLP Processor
#'
#' @description
#' Internal function to get the initialized spaCy processor.
#'
#' @param model Model to use if not initialized
#' @return The SpacyNLP Python object
#' @keywords internal
get_spacy_nlp <- function(model = "en_core_web_sm") {
  if (!exists("nlp", envir = .spacy_env)) {
    init_spacy_nlp(model)
  }
  get("nlp", envir = .spacy_env)
}


#' Parse Texts with Full spaCy Features
#'
#' @description
#' Parse texts using spaCy and extract all linguistic features including
#' morphology, dependency parsing, POS tags, lemmas, and named entities.
#'
#' @param texts Character vector of texts to parse.
#' @param include_pos Logical; include part-of-speech tags (default: TRUE).
#' @param include_lemma Logical; include lemmatized forms (default: TRUE).
#' @param include_entity Logical; include named entity recognition (default: TRUE).
#' @param include_dependency Logical; include dependency parsing (default: TRUE).
#' @param include_morphology Logical; include morphological features (default: TRUE).
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data.frame with token-level annotations:
#'   \itemize{
#'     \item \code{doc_id}: Document identifier
#'     \item \code{sentence_id}: Sentence number within document
#'     \item \code{token_id}: Token position within sentence
#'     \item \code{token}: Original token text
#'     \item \code{pos}: Universal POS tag (NOUN, VERB, ADJ, etc.)
#'     \item \code{tag}: Fine-grained POS tag (NN, VBD, JJ, etc.)
#'     \item \code{lemma}: Lemmatized form
#'     \item \code{entity}: Named entity type (PERSON, ORG, GPE, etc.)
#'     \item \code{entity_iob}: IOB tag (B=beginning, I=inside, O=outside)
#'     \item \code{dep_rel}: Dependency relation (nsubj, dobj, amod, etc.)
#'     \item \code{head_token_id}: Head token in dependency tree
#'     \item \code{morph}: Full morphological features string
#'     \item \code{morph_*}: Individual morphological features as columns
#'   }
#'
#' @details
#' This function uses Python spaCy via reticulate, providing access to
#' features not available in spacyr, including morphological analysis.
#'
#' Morphological features include:
#' \itemize{
#'   \item \code{morph_Number}: Sing, Plur
#'   \item \code{morph_Person}: 1, 2, 3
#'   \item \code{morph_Tense}: Past, Pres, Fut
#'   \item \code{morph_VerbForm}: Fin, Inf, Part, Ger
#'   \item \code{morph_Mood}: Ind, Imp, Sub
#'   \item \code{morph_Case}: Nom, Acc, Dat, Gen
#'   \item And more depending on the language model
#' }
#'
#' @family nlp
#' @export
#'
#' @examples
#' \dontrun{
#' texts <- c(
#'   "Apple Inc. was founded by Steve Jobs.",
#'   "The cats are sleeping on the couch."
#' )
#' result <- spacy_parse_full(texts)
#'
#' # View morphological features
#' result[, c("token", "pos", "morph")]
#'
#' # Filter by POS
#' verbs <- result[result$pos == "VERB", ]
#' print(verbs[, c("token", "lemma", "morph_Tense", "morph_VerbForm")])
#' }
spacy_parse_full <- function(texts,
                              include_pos = TRUE,
                              include_lemma = TRUE,
                              include_entity = TRUE,
                              include_dependency = TRUE,
                              include_morphology = TRUE,
                              model = "en_core_web_sm") {

  nlp <- get_spacy_nlp(model)

  result <- nlp$parse_to_dataframe(
    texts = as.list(texts),
    include_pos = include_pos,
    include_lemma = include_lemma,
    include_entity = include_entity,
    include_dependency = include_dependency,
    include_morphology = include_morphology
  )

  # Convert list of dicts to data.frame
  if (length(result) == 0) {
    return(data.frame())
  }

  # Get all unique column names across all rows
  all_cols <- unique(unlist(lapply(result, names)))

  # Convert to data.frame, handling missing columns
  df <- do.call(rbind, lapply(result, function(row) {
    # Fill missing columns with NA
    for (col in all_cols) {
      if (!(col %in% names(row))) {
        row[[col]] <- NA
      }
    }
    as.data.frame(row[all_cols], stringsAsFactors = FALSE)
  }))

  # Ensure consistent column order
  base_cols <- c("doc_id", "sentence_id", "token_id", "token")
  if (include_pos) base_cols <- c(base_cols, "pos", "tag")
  if (include_lemma) base_cols <- c(base_cols, "lemma")
  if (include_entity) base_cols <- c(base_cols, "entity", "entity_iob")
  if (include_dependency) base_cols <- c(base_cols, "dep_rel", "head_token_id")
  if (include_morphology) base_cols <- c(base_cols, "morph")

  # Add morph columns at the end
  morph_cols <- grep("^morph_", names(df), value = TRUE)
  final_cols <- c(intersect(base_cols, names(df)), morph_cols)

  df[, final_cols, drop = FALSE]
}


#' Extract Named Entities
#'
#' @description
#' Extract named entities at the span level using spaCy.
#'
#' @param texts Character vector of texts.
#' @param model Character; spaCy model to use.
#'
#' @return A data.frame with columns:
#'   \itemize{
#'     \item \code{doc_id}: Document identifier
#'     \item \code{text}: Entity text
#'     \item \code{label}: Entity type (PERSON, ORG, GPE, DATE, etc.)
#'     \item \code{start_char}: Start character position
#'     \item \code{end_char}: End character position
#'   }
#'
#' @family nlp
#' @export
#'
#' @examples
#' \dontrun{
#' texts <- c("Apple was founded by Steve Jobs in Cupertino, California.")
#' entities <- spacy_extract_entities(texts)
#' print(entities)
#' }
spacy_extract_entities <- function(texts, model = "en_core_web_sm") {

  nlp <- get_spacy_nlp(model)
  result <- nlp$extract_entities(as.list(texts))

  if (length(result) == 0) {
    return(data.frame(
      doc_id = character(),
      text = character(),
      label = character(),
      start_char = integer(),
      end_char = integer()
    ))
  }

  do.call(rbind, lapply(result, as.data.frame, stringsAsFactors = FALSE))
}


#' Extract Noun Chunks
#'
#' @description
#' Extract noun chunks (base noun phrases) using spaCy.
#'
#' @param texts Character vector of texts.
#' @param model Character; spaCy model to use.
#'
#' @return A data.frame with noun chunk information.
#'
#' @family nlp
#' @export
#'
#' @examples
#' \dontrun{
#' texts <- c("The quick brown fox jumps over the lazy dog.")
#' chunks <- spacy_extract_noun_chunks(texts)
#' print(chunks)
#' }
spacy_extract_noun_chunks <- function(texts, model = "en_core_web_sm") {

  nlp <- get_spacy_nlp(model)
  result <- nlp$extract_noun_chunks(as.list(texts))

  if (length(result) == 0) {
    return(data.frame(
      doc_id = character(),
      text = character(),
      root_text = character(),
      root_pos = character(),
      root_dep = character()
    ))
  }

  do.call(rbind, lapply(result, as.data.frame, stringsAsFactors = FALSE))
}


#' Get spaCy Model Information
#'
#' @description
#' Get information about the loaded spaCy model.
#'
#' @param model Character; spaCy model to query.
#'
#' @return A list with model metadata including:
#'   \itemize{
#'     \item \code{model_name}: Model identifier
#'     \item \code{lang}: Language code
#'     \item \code{pipeline}: List of pipeline components
#'     \item \code{has_vectors}: Whether model includes word vectors
#'     \item \code{vector_dim}: Dimension of word vectors (if available)
#'   }
#'
#' @family nlp
#' @export
#'
#' @examples
#' \dontrun{
#' info <- spacy_model_info()
#' print(info)
#' }
spacy_model_info <- function(model = "en_core_web_sm") {
  nlp <- get_spacy_nlp(model)
  nlp$get_model_info()
}


#' Calculate Text Similarity
#'
#' @description
#' Calculate semantic similarity between two texts using spaCy word vectors.
#' Requires a model with word vectors (en_core_web_md or en_core_web_lg).
#'
#' @param text1 First text string.
#' @param text2 Second text string.
#' @param model Character; spaCy model to use (must have word vectors).
#'
#' @return Numeric similarity score between 0 and 1.
#'
#' @family nlp
#' @export
#'
#' @examples
#' \dontrun{
#' # Requires medium or large model
#' init_spacy_nlp("en_core_web_md")
#' sim <- spacy_similarity("I love dogs", "I adore puppies")
#' print(sim)  # High similarity
#'
#' sim2 <- spacy_similarity("I love dogs", "The weather is nice")
#' print(sim2)  # Low similarity
#' }
spacy_similarity <- function(text1, text2, model = "en_core_web_md") {
  nlp <- get_spacy_nlp(model)
  nlp$calculate_similarity(text1, text2)
}
