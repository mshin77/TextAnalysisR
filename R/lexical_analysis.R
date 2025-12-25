#' @title Lexical Analysis Functions
#'
#' @description
#' Comprehensive functions for lexical analysis including:
#' - Linguistic Annotation (POS tagging, NER)
#' - Frequency Analysis (word frequency, n-grams, MWEs)
#' - Keywords (TF-IDF, keyness)
#' - Lexical Diversity (TTR, MTLD, MATTR)
#' - Readability (Flesch, Gunning Fog, etc.)
#'
#' @name lexical_analysis
#' @family lexical
NULL


.lexdiv_cache <- new.env(hash = TRUE, parent = emptyenv())

#' Clear Lexical Diversity Cache
#'
#' @description
#' Clears the internal cache used for lexical diversity calculations.
#' Call this function if you need to free memory or ensure fresh calculations.
#'
#' @return Invisible NULL
#' @family lexical
#' @export
clear_lexdiv_cache <- function() {
  rm(list = ls(.lexdiv_cache), envir = .lexdiv_cache)
  invisible(NULL)
}


#' @title Detect Multi-Word Expressions
#'
#' @description
#' This function detects multi-word expressions (collocations) of specified
#' sizes that appear at least a specified number of times in the provided tokens.
#'
#' @param tokens A \code{tokens} object from the \code{quanteda} package.
#' @param size A numeric vector specifying the sizes of the collocations to detect (default: 2:5).
#' @param min_count The minimum number of occurrences for a collocation to be
#'   considered (default: 2).
#'
#' @return A character vector of detected collocations.
#'
#' @family lexical
#' @export
#'
#' @examples
#' if (interactive()) {
#'   mydata <- TextAnalysisR::SpecialEduTech
#'
#'   united_tbl <- TextAnalysisR::unite_cols(
#'     mydata,
#'     listed_vars = c("title", "keyword", "abstract")
#'   )
#'
#'   tokens <- TextAnalysisR::prep_texts(united_tbl, text_field = "united_texts")
#'
#'   collocations <- TextAnalysisR::detect_multi_words(tokens, size = 2:5, min_count = 2)
#'   print(collocations)
#' }
detect_multi_words <- function(tokens, size = 2:5, min_count = 2) {
  tstat <- quanteda.textstats::textstat_collocations(tokens, size = size, min_count = min_count)
  tstat_collocation <- tstat$collocation
  return(tstat_collocation)
}


#' Extract Part-of-Speech Tags from Tokens
#'
#' @description
#' Uses spaCy to extract part-of-speech (POS) tags from tokenized text.
#' Returns a data frame with token-level POS annotations.
#'
#' @param tokens A quanteda tokens object or character vector of texts.
#' @param include_lemma Logical; include lemmatized forms (default: TRUE).
#' @param include_entity Logical; include named entity recognition (default: FALSE).
#' @param include_dependency Logical; include dependency parsing (default: FALSE).
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data frame with columns:
#' \itemize{
#'   \item \code{doc_id}: Document identifier
#'   \item \code{sentence_id}: Sentence number within document
#'   \item \code{token_id}: Token position within sentence
#'   \item \code{token}: Original token
#'   \item \code{pos}: Universal POS tag (e.g., NOUN, VERB, ADJ)
#'   \item \code{tag}: Detailed POS tag (e.g., NN, VBD, JJ)
#'   \item \code{lemma}: Lemmatized form (if include_lemma = TRUE)
#'   \item \code{entity}: Named entity type (if include_entity = TRUE)
#'   \item \code{head_token_id}: Head token in dependency tree (if include_dependency = TRUE)
#'   \item \code{dep_rel}: Dependency relation type, e.g., nsubj, dobj (if include_dependency = TRUE)
#' }
#'
#' @details
#' This function requires the spacyr package and a working Python environment
#' with spaCy installed. If spaCy is not initialized, this function will
#' attempt to initialize it with the specified model.
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' tokens <- quanteda::tokens("The quick brown fox jumps over the lazy dog.")
#' pos_data <- extract_pos_tags(tokens)
#' print(pos_data)
#' }
extract_pos_tags <- function(tokens,
                             include_lemma = TRUE,
                             include_entity = FALSE,
                             include_dependency = FALSE,
                             model = "en_core_web_sm") {

  if (!requireNamespace("spacyr", quietly = TRUE)) {
    stop("Package 'spacyr' is required for POS tagging. Please install it with: install.packages('spacyr')")
  }

  # Convert tokens to text if needed
  if (inherits(tokens, "tokens")) {
    texts <- sapply(tokens, function(x) paste(x, collapse = " "))
  } else if (is.character(tokens)) {
    texts <- tokens
  } else {
    stop("tokens must be a quanteda tokens object or character vector")
  }

  # Try to initialize spaCy if not already done
  tryCatch({
    # Check if spaCy is initialized by trying a simple operation
    spacyr::spacy_parse("test", pos = FALSE, tag = FALSE, lemma = FALSE, entity = FALSE)
  }, error = function(e) {
    message("Initializing spaCy with model: ", model)
    spacyr::spacy_initialize(model = model)
  })

  # Parse with POS and tag enabled
  parsed <- spacyr::spacy_parse(
    texts,
    pos = TRUE,
    tag = TRUE,
    lemma = include_lemma,
    entity = include_entity,
    dependency = include_dependency
  )

  return(parsed)
}


#' Extract Morphological Features
#'
#' @description
#' Uses spaCy to extract comprehensive morphological features from text.
#' Returns data with Number, Tense, VerbForm, Person, Case, Mood, Aspect, etc.
#'
#' @param tokens A quanteda tokens object or character vector of texts.
#' @param features Character vector of morphological features to extract.
#'   Default includes common Universal Dependencies features.
#' @param include_pos Logical; include POS tags (default: TRUE).
#' @param include_lemma Logical; include lemmatized forms (default: TRUE).
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data frame with token-level morphological annotations including
#'   morph_* columns for each requested feature.
#'
#' @details
#' Morphological features follow Universal Dependencies annotation.
#' Common features include:
#' \itemize{
#'   \item \code{Number}: Sing (singular), Plur (plural)
#'   \item \code{Tense}: Past, Pres (present), Fut (future)
#'   \item \code{VerbForm}: Fin (finite), Inf (infinitive), Part (participle), Ger (gerund)
#'   \item \code{Person}: 1, 2, 3 (first, second, third person)
#'   \item \code{Case}: Nom (nominative), Acc (accusative), Gen (genitive), Dat (dative)
#'   \item \code{Mood}: Ind (indicative), Imp (imperative), Sub (subjunctive)
#'   \item \code{Aspect}: Perf (perfective), Imp (imperfective), Prog (progressive)
#' }
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' tokens <- quanteda::tokens("The cats are running quickly.")
#' morph_data <- extract_morphology(tokens)
#' print(morph_data)
#' }
extract_morphology <- function(tokens,
                               features = c("Number", "Tense", "VerbForm",
                                            "Person", "Case", "Mood", "Aspect"),
                               include_pos = TRUE,
                               include_lemma = TRUE,
                               model = "en_core_web_sm") {

  if (!requireNamespace("spacyr", quietly = TRUE)) {
    stop("Package 'spacyr' is required. Install with: install.packages('spacyr')")
  }

  # Convert tokens to text if needed
  if (inherits(tokens, "tokens")) {
    texts <- sapply(tokens, function(x) paste(x, collapse = " "))
  } else if (is.character(tokens)) {
    texts <- tokens
  } else {
    stop("tokens must be a quanteda tokens object or character vector")
  }

  # Initialize spaCy if needed
  tryCatch({
    spacyr::spacy_parse("test", pos = FALSE, lemma = FALSE, entity = FALSE)
  }, error = function(e) {
    message("Initializing spaCy with model: ", model)
    spacyr::spacy_initialize(model = model)
  })

  # Parse with morphology extraction via additional_attributes
  parsed <- spacyr::spacy_parse(
    texts,
    pos = include_pos,
    tag = include_pos,
    lemma = include_lemma,
    entity = FALSE,
    dependency = FALSE,
    additional_attributes = c("morph")
  )

  # Parse the morph string into individual feature columns
  if ("morph" %in% names(parsed) && nrow(parsed) > 0) {
    parsed <- parse_morphology_string(parsed, features)
  }

  return(parsed)
}


#' Parse Morphology String into Individual Columns
#'
#' @description
#' Parse spaCy's morphology string format
#' (e.g., "Number=Sing|Tense=Past|VerbForm=Fin") into individual columns.
#'
#' @param parsed Data frame with a 'morph' column from spaCy.
#' @param features Character vector of feature names to extract.
#'
#' @return Data frame with additional morph_* columns for each feature.
#'
#' @keywords internal
parse_morphology_string <- function(parsed, features) {
  for (feat in features) {
    col_name <- paste0("morph_", feat)
    parsed[[col_name]] <- vapply(parsed$morph, function(m) {
      # Robust check for empty/NA/NULL values
      if (is.null(m) || length(m) == 0 || !is.atomic(m)) return(NA_character_)
      m_str <- as.character(m)[1]
      if (is.na(m_str) || m_str == "") return(NA_character_)
      # Parse "Number=Sing|Tense=Past|..." format
      parts <- strsplit(m_str, "\\|")[[1]]
      for (part in parts) {
        kv <- strsplit(part, "=")[[1]]
        if (length(kv) == 2 && kv[1] == feat) {
          return(kv[2])
        }
      }
      return(NA_character_)
    }, FUN.VALUE = character(1), USE.NAMES = FALSE)
  }
  return(parsed)
}


#' Plot Morphology Feature Distribution
#'
#' @description
#' Creates a bar chart showing the distribution of a morphological feature
#' using consistent package styling.
#'
#' @param data Data frame with morph_* columns from extract_morphology().
#' @param feature Character; feature name (e.g., "Number", "Tense").
#' @param title Character; plot title (auto-generated if NULL).
#' @param colors Named character vector of custom colors for feature values.
#'
#' @return A plotly object.
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' morph_data <- extract_morphology(tokens)
#' plot_morphology_feature(morph_data, "Tense")
#' }
plot_morphology_feature <- function(data,
                                    feature,
                                    title = NULL,
                                    colors = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required for visualization.")
  }

  col_name <- paste0("morph_", feature)

  if (!col_name %in% names(data)) {
    # Return empty plot with message
    return(
      plotly::plot_ly() %>%
        plotly::layout(
          title = list(text = paste("Feature", feature, "not available"),
                       font = list(size = 14, color = "#6B7280")),
          xaxis = list(visible = FALSE),
          yaxis = list(visible = FALSE),
          annotations = list(
            list(text = "No data", showarrow = FALSE, font = list(size = 16))
          )
        )
    )
  }

  values <- data[[col_name]]
  values <- values[!is.na(values) & values != ""]

  if (length(values) == 0) {
    return(
      plotly::plot_ly() %>%
        plotly::layout(
          title = list(text = paste("No", feature, "data found"),
                       font = list(size = 14, color = "#6B7280"))
        )
    )
  }

  freq_df <- as.data.frame(table(values), stringsAsFactors = FALSE)
  names(freq_df) <- c("Value", "Count")
  freq_df <- freq_df[order(-freq_df$Count), ]
  freq_df$Percentage <- round(freq_df$Count / sum(freq_df$Count) * 100, 1)

  if (is.null(title)) {
    title <- paste(feature, "Distribution")
  }

  # Feature-specific default colors
  if (is.null(colors)) {
    colors <- switch(feature,
      "Number" = c("Sing" = "#3B82F6", "Plur" = "#10B981"),
      "Tense" = c("Past" = "#EF4444", "Pres" = "#3B82F6", "Fut" = "#10B981"),
      "VerbForm" = c("Fin" = "#3B82F6", "Inf" = "#8B5CF6",
                     "Part" = "#F59E0B", "Ger" = "#10B981"),
      "Person" = c("1" = "#3B82F6", "2" = "#10B981", "3" = "#F59E0B"),
      "Case" = c("Nom" = "#3B82F6", "Acc" = "#10B981",
                 "Gen" = "#F59E0B", "Dat" = "#8B5CF6"),
      "Mood" = c("Ind" = "#3B82F6", "Imp" = "#EF4444", "Sub" = "#8B5CF6"),
      "Aspect" = c("Perf" = "#3B82F6", "Imp" = "#10B981", "Prog" = "#F59E0B"),
      NULL
    )
  }

  bar_colors <- if (!is.null(colors) && length(colors) > 0) {
    sapply(freq_df$Value, function(v) {
      if (v %in% names(colors)) colors[[v]] else "#6B7280"
    })
  } else {
    rep("#337ab7", nrow(freq_df))
  }

  plotly::plot_ly(
    data = freq_df,
    x = ~Value,
    y = ~Count,
    type = "bar",
    marker = list(color = bar_colors),
    hoverinfo = "text",
    hovertext = ~paste0(Value, "\nCount: ", Count, "\n", Percentage, "%")
  ) %>%
    plotly::layout(
      title = list(text = title, font = list(size = 14, color = "#0c1f4a")),
      xaxis = list(title = "", tickfont = list(size = 12)),
      yaxis = list(title = "Frequency", titlefont = list(size = 12)),
      margin = list(t = 40, b = 40, l = 50, r = 20)
    )
}


#' Summarize Morphology Features
#'
#' @description
#' Creates a summary table of morphological feature distributions
#' with counts and percentages for each feature value.
#'
#' @param data Data frame with morph_* columns from extract_morphology().
#' @param features Character vector of features to summarize.
#'   If NULL, all available morph_* columns are used.
#'
#' @return A data frame with Feature, Value, Count, and Percentage columns.
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' morph_data <- extract_morphology(tokens)
#' summary_df <- summarize_morphology(morph_data)
#' print(summary_df)
#' }
summarize_morphology <- function(data, features = NULL) {
  morph_cols <- grep("^morph_", names(data), value = TRUE)

  if (length(morph_cols) == 0) {
    return(data.frame(
      Feature = character(0),
      Value = character(0),
      Count = integer(0),
      Percentage = numeric(0),
      stringsAsFactors = FALSE
    ))
  }

  if (!is.null(features)) {
    target_cols <- paste0("morph_", features)
    morph_cols <- intersect(morph_cols, target_cols)
  }

  summary_list <- lapply(morph_cols, function(col) {
    feat_name <- gsub("morph_", "", col)
    values <- data[[col]]
    values <- values[!is.na(values) & values != ""]

    if (length(values) == 0) return(NULL)

    counts <- as.data.frame(table(values), stringsAsFactors = FALSE)
    names(counts) <- c("Value", "Count")
    counts$Feature <- feat_name
    counts$Percentage <- round(counts$Count / sum(counts$Count) * 100, 1)
    counts[, c("Feature", "Value", "Count", "Percentage")]
  })

  result <- do.call(rbind, summary_list)
  if (is.null(result)) {
    return(data.frame(
      Feature = character(0),
      Value = character(0),
      Count = integer(0),
      Percentage = numeric(0),
      stringsAsFactors = FALSE
    ))
  }

  return(result)
}


#' Extract Named Entities from Tokens
#'
#' @description
#' Uses spaCy to extract named entities (NER) from tokenized text.
#' Returns a data frame with token-level entity annotations.
#'
#' @param tokens A quanteda tokens object or character vector of texts.
#' @param include_pos Logical; include POS tags (default: TRUE).
#' @param include_lemma Logical; include lemmatized forms (default: TRUE).
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data frame with columns:
#' \itemize{
#'   \item \code{doc_id}: Document identifier
#'   \item \code{token}: Original token
#'   \item \code{entity}: Named entity type (e.g., PERSON, ORG, GPE)
#'   \item \code{pos}: Universal POS tag (if include_pos = TRUE)
#'   \item \code{lemma}: Lemmatized form (if include_lemma = TRUE)
#' }
#'
#' @details
#' This function requires the spacyr package and a working Python environment
#' with spaCy installed. If spaCy is not initialized, this function will
#' attempt to initialize it with the specified model.
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' tokens <- quanteda::tokens("Apple Inc. was founded by Steve Jobs in California.")
#' entity_data <- extract_named_entities(tokens)
#' print(entity_data)
#' }
extract_named_entities <- function(tokens,
                                   include_pos = TRUE,
                                   include_lemma = TRUE,
                                   model = "en_core_web_sm") {

  if (!requireNamespace("spacyr", quietly = TRUE)) {
    stop("Package 'spacyr' is required for NER. Please install it with: install.packages('spacyr')")
  }

  # Convert tokens to text if needed
  if (inherits(tokens, "tokens")) {
    texts <- sapply(tokens, function(x) paste(x, collapse = " "))
  } else if (is.character(tokens)) {
    texts <- tokens
  } else {
    stop("tokens must be a quanteda tokens object or character vector")
  }

  # Try to initialize spaCy if not already done
  tryCatch({
    spacyr::spacy_parse("test", pos = FALSE, tag = FALSE, lemma = FALSE, entity = FALSE)
  }, error = function(e) {
    message("Initializing spaCy with model: ", model)
    spacyr::spacy_initialize(model = model)
  })

  # Parse with entity enabled

  parsed <- spacyr::spacy_parse(
    texts,
    pos = include_pos,
    tag = include_pos,
    lemma = include_lemma,
    entity = TRUE,
    dependency = FALSE
  )

  return(parsed)
}


#' Lexical Diversity Analysis
#'
#' @description
#' Calculates multiple lexical diversity metrics for a document-feature matrix (DFM)
#' or tokens object. Supports all quanteda.textstats measures plus MTLD
#' (Measure of Textual Lexical Diversity), which is the most recommended measure
#' according to McCarthy & Jarvis (2010) for being independent of text length.
#'
#' @param x A quanteda DFM or tokens object. Tokens object is preferred for
#'   accurate MTLD calculation since it preserves token order.
#' @param measures Character vector of measures to calculate.
#'   Default is "all" which includes: TTR, C, R, CTTR, U, S, K, I, D, Vm, Maas, MATTR, MSTTR, and MTLD.
#'   Most recommended: "MTLD" or "MATTR" for length-independent measures.
#' @param texts Optional character vector of original texts. Required for MTLD
#'   calculation when using DFM input (since DFM loses token order).
#' @param cache_key Optional cache key (e.g., from digest::digest) for caching
#'   expensive calculations. Use the same cache_key to retrieve cached results.
#'
#' @return A list containing:
#' \itemize{
#'   \item \code{lexical_diversity}: Data frame with per-document lexical diversity scores
#'   \item \code{summary_stats}: List of summary statistics (mean, median, sd) for each measure
#' }
#'
#' @details
#' MTLD (Measure of Textual Lexical Diversity) is calculated using the algorithm

#' from McCarthy & Jarvis (2010). It counts the number of "factors" needed to
#' reduce TTR below 0.72, then divides the number of tokens by the number of factors.
#' This provides a length-independent measure of lexical diversity.
#'
#' Important notes:
#' \itemize{
#'   \item For MTLD accuracy, pass a tokens object (not DFM) as input
#'   \item If using DFM, provide the 'texts' parameter for MTLD calculation
#'   \item MATTR and MSTTR window sizes are automatically adjusted for short documents
#'   \item Results are cached when cache_key is provided for repeated analysis
#' }
#'
#' @references
#' McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study
#' of sophisticated approaches to lexical diversity assessment.
#' Behavior Research Methods, 42(2), 381-392.
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' data(SpecialEduTech)
#' texts <- SpecialEduTech$abstract[1:10]
#' corp <- quanteda::corpus(texts)
#' toks <- quanteda::tokens(corp)
#' # Preferred: pass tokens object for accurate MTLD
#' lex_div <- lexical_diversity_analysis(toks, texts = texts)
#' # With caching for repeated analysis
#' cache_key <- digest::digest(texts)
#' lex_div <- lexical_diversity_analysis(toks, texts = texts, cache_key = cache_key)
#' # Alternative: pass DFM with texts for MTLD accuracy
#' dfm_obj <- quanteda::dfm(toks)
#' lex_div <- lexical_diversity_analysis(dfm_obj, texts = texts)
#' print(lex_div)
#' }
#'
#' @importFrom quanteda.textstats textstat_lexdiv
#' @importFrom quanteda docnames
lexical_diversity_analysis <- function(x,
                                      measures = "all",
                                      texts = NULL,
                                      cache_key = NULL) {

  # Check cache first if cache_key is provided

  if (!is.null(cache_key) && nzchar(cache_key)) {
    cache_id <- paste0("lexdiv_", cache_key)
    if (exists(cache_id, envir = .lexdiv_cache, inherits = FALSE)) {
      return(get(cache_id, envir = .lexdiv_cache, inherits = FALSE))
    }
  }

  if (!requireNamespace("quanteda.textstats", quietly = TRUE)) {
    stop("Package 'quanteda.textstats' is required. Please install it.")
  }

  # Measures available in quanteda.textstats

  quanteda_measures <- c("TTR", "C", "R", "CTTR", "U", "S", "K", "I", "D", "Vm", "Maas", "MATTR", "MSTTR")

  # MTLD requires koRpus package (most recommended measure per McCarthy & Jarvis 2010)
  mtld_requested <- FALSE

  if ("all" %in% measures) {
    measures_to_use <- quanteda_measures
    mtld_requested <- TRUE
  } else {
    if ("MTLD" %in% measures) {
      mtld_requested <- TRUE
      measures <- setdiff(measures, "MTLD")
    }
    measures_to_use <- intersect(measures, quanteda_measures)
  }

  # Determine input type and prepare tokens for MTLD if needed
  is_tokens_input <- inherits(x, "tokens")

  # For MTLD with DFM input, we need proper token sequences
  # Create tokens from texts if available, otherwise warn
  mtld_tokens <- NULL
  if (mtld_requested && !is_tokens_input) {
    if (!is.null(texts) && length(texts) == quanteda::ndoc(x)) {
      # Create tokens from original texts to preserve order for MTLD
      mtld_tokens <- quanteda::tokens(texts, remove_punct = TRUE)
    } else {
      message("Note: MTLD requires sequential token order (McCarthy & Jarvis, 2010). ",
              "DFM input loses token order. For accurate MTLD, pass a tokens object ",
              "or provide the 'texts' parameter. Skipping MTLD calculation.")
      mtld_requested <- FALSE
    }
  }

  tryCatch({
    # Calculate minimum document length to set appropriate window size
    doc_lengths <- quanteda::ntoken(x)
    min_length <- min(doc_lengths)

    # Set window size for MATTR/MSTTR (default is 100, adjust if documents are shorter)
    window_size <- min(100, max(10, min_length))

    # Calculate quanteda measures if any requested
    if (length(measures_to_use) > 0) {
      lexdiv_results <- suppressWarnings(
        quanteda.textstats::textstat_lexdiv(
          x,
          measure = measures_to_use,
          MATTR_window = window_size,
          MSTTR_segment = window_size
        )
      )
    } else {
      # Create empty data frame with document names
      lexdiv_results <- data.frame(document = quanteda::docnames(x))
    }

    # Add MTLD if requested - use custom implementation (McCarthy & Jarvis 2010)
    if (mtld_requested) {
      tryCatch({
        # Optimized MTLD implementation based on McCarthy & Jarvis (2010)
        # Uses O(n) environment-based hash tracking instead of O(n^2) vector concatenation
        calculate_mtld <- function(tokens, factor_size = 0.72) {
          n <- length(tokens)
          if (n < 10) return(NA_real_)

          # Helper function to calculate MTLD in one direction using O(n) algorithm
          mtld_one_direction <- function(toks) {
            n_toks <- length(toks)
            seen <- new.env(hash = TRUE, size = n_toks)
            unique_count <- 0
            factors <- 0
            start_idx <- 1

            for (i in seq_len(n_toks)) {
              token <- toks[i]
              if (!exists(token, envir = seen, inherits = FALSE)) {
                assign(token, TRUE, envir = seen)
                unique_count <- unique_count + 1
              }
              current_length <- i - start_idx + 1
              current_ttr <- unique_count / current_length

              if (current_ttr <= factor_size) {
                factors <- factors + 1
                # Reset for new factor
                rm(list = ls(seen), envir = seen)
                unique_count <- 0
                start_idx <- i + 1
              }
            }

            # Add partial factor for remaining tokens
            if (start_idx <= n_toks) {
              remaining_length <- n_toks - start_idx + 1
              if (remaining_length > 0 && unique_count > 0) {
                final_ttr <- unique_count / remaining_length
                partial <- (1 - final_ttr) / (1 - factor_size)
                factors <- factors + partial
              }
            }

            if (factors > 0) n_toks / factors else NA_real_
          }

          # Forward and backward MTLD
          forward_mtld <- mtld_one_direction(tokens)
          backward_mtld <- mtld_one_direction(rev(tokens))

          # Average forward and backward
          mean(c(forward_mtld, backward_mtld), na.rm = TRUE)
        }

        # Determine token source for MTLD calculation
        tokens_for_mtld <- if (is_tokens_input) x else mtld_tokens

        # Get tokens for each document
        mtld_values <- sapply(seq_len(quanteda::ndoc(tokens_for_mtld)), function(i) {
          doc_tokens <- as.character(tokens_for_mtld[[i]])
          if (length(doc_tokens) < 10) return(NA_real_)
          calculate_mtld(doc_tokens)
        })

        lexdiv_results$MTLD <- as.numeric(mtld_values)
      }, error = function(e) {
        message("MTLD calculation failed: ", e$message, ". Skipping MTLD.")
      })
    }

    # Add document names if not present
    if (!"document" %in% names(lexdiv_results)) {
      lexdiv_results$document <- quanteda::docnames(x)
    }

    # Standardize document names to "Doc 1, Doc 2..." format
    lexdiv_results$document <- paste0("Doc ", seq_len(nrow(lexdiv_results)))

    # Get actual column names from result (after MTLD is added)
    actual_measures <- setdiff(names(lexdiv_results), "document")

    # Select only columns that exist
    cols_to_keep <- c("document", actual_measures)
    lexdiv_results <- lexdiv_results[, cols_to_keep, drop = FALSE]

    # Add Average Sentence Length if texts provided
    if (!is.null(texts) && length(texts) == nrow(lexdiv_results)) {
      avg_sentence_length <- sapply(texts, function(t) {
        sents <- unlist(strsplit(t, "[.!?]+"))
        sents <- sents[nzchar(trimws(sents))]
        words <- unlist(strsplit(paste(sents, collapse = " "), "\\s+"))
        if (length(sents) == 0) return(NA)
        length(words) / length(sents)
      })
      lexdiv_results$`Avg Sentence Length` <- avg_sentence_length
      actual_measures <- c(actual_measures, "Avg Sentence Length")
    }

    summary_stats <- list(
      n_documents = nrow(lexdiv_results),
      measures_calculated = actual_measures
    )

    for (measure in actual_measures) {
      if (measure %in% names(lexdiv_results)) {
        summary_stats[[paste0(measure, "_mean")]] <- mean(lexdiv_results[[measure]], na.rm = TRUE)
        summary_stats[[paste0(measure, "_median")]] <- median(lexdiv_results[[measure]], na.rm = TRUE)
        summary_stats[[paste0(measure, "_sd")]] <- sd(lexdiv_results[[measure]], na.rm = TRUE)
      }
    }

    result <- list(
      lexical_diversity = lexdiv_results,
      summary_stats = summary_stats
    )

    # Store in cache if cache_key provided
    if (!is.null(cache_key) && nzchar(cache_key)) {
      cache_id <- paste0("lexdiv_", cache_key)
      assign(cache_id, result, envir = .lexdiv_cache)
    }

    return(result)

  }, error = function(e) {
    stop("Error calculating lexical diversity: ", e$message)
  })
}


#' Plot Lexical Diversity Distribution
#'
#' @description
#' Creates a boxplot showing the distribution of a lexical diversity metric.
#'
#' @param lexdiv_data Data frame from lexical_diversity_analysis()
#' @param metric Metric to plot. Recommended: "MTLD" or "MATTR" (text-length independent)
#' @param title Plot title (default: auto-generated)
#'
#' @return A plotly boxplot
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' data(SpecialEduTech)
#' texts <- SpecialEduTech$abstract[1:10]
#' corp <- quanteda::corpus(texts)
#' toks <- quanteda::tokens(corp)
#' dfm_obj <- quanteda::dfm(toks)
#' result <- lexical_diversity_analysis(dfm_obj)
#' plot <- plot_lexical_diversity_distribution(result$lexical_diversity, "MTLD")
#' print(plot)
#' }
plot_lexical_diversity_distribution <- function(lexdiv_data,
                                               metric,
                                               title = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required.")
  }

  if (!metric %in% names(lexdiv_data)) {
    stop(paste("Metric", metric, "not found in lexical diversity data"))
  }

  if (is.null(title)) {
    title <- paste(metric, "- Overall Distribution")
  }

  metric_values <- lexdiv_data[[metric]]
  metric_values <- metric_values[is.finite(metric_values)]

  if (length(metric_values) == 0) {
    return(plot_error("No valid data for selected metric"))
  }

  plotly::plot_ly(
    y = metric_values,
    type = "box",
    name = "Distribution",
    marker = list(color = "#8B5CF6"),
    line = list(color = "#0c1f4a"),
    fillcolor = "rgba(139, 92, 246, 0.7)",
    hoverinfo = "y",
    hoverlabel = get_plotly_hover_config()
  ) %>%
    apply_standard_plotly_layout(
      title = title,
      yaxis_title = metric,
      margin = list(t = 60, b = 60, l = 80, r = 40)
    )

}


#' @title Lexical Frequency Analysis
#'
#' @description
#' Wrapper function for plot_word_frequency for lexical analysis.
#'
#' @param ... Arguments passed to plot_word_frequency
#'
#' @family lexical
#' @export
lexical_frequency_analysis <- function(...) {
  return(plot_word_frequency(...))
}



#' @title Plot Word Frequency
#'
#' @description
#' Creates a bar plot showing the most frequent words in a document-feature matrix (dfm).
#'
#' @param dfm_object A document-feature matrix created by quanteda::dfm().
#' @param n The number of top words to display (default: 20).
#' @param height The height of the resulting Plotly plot, in pixels (default: 800).
#' @param width The width of the resulting Plotly plot, in pixels (default: 1000).
#' @param ... Additional arguments passed to plotly::ggplotly().
#'
#' @return A plotly object showing word frequency.
#'
#' @family visualization
#' @export
#'
#' @examples
#' if (interactive()) {
#'   texts <- c("mathematics technology", "education technology", "learning support")
#'   dfm <- quanteda::dfm(quanteda::tokens(texts))
#'   plot <- plot_word_frequency(dfm, n = 5)
#'   print(plot)
#' }
plot_word_frequency <- function(dfm_object,
                                n = 20,
                                height = NULL,
                                width = NULL,
                                ...) {

  if (!inherits(dfm_object, "dfm")) {
    stop("Input must be a quanteda dfm object")
  }

  freq_df <- quanteda.textstats::textstat_frequency(dfm_object, n = n) %>%
    dplyr::mutate(
      feature = stats::reorder(feature, frequency)
    )

  ggplot_obj <- ggplot2::ggplot(freq_df,
                                ggplot2::aes(x = feature, y = frequency,
                                            text = paste("Word:", feature,
                                                       "<br>Frequency:", frequency))) +
    ggplot2::geom_point(color = "#0c1f4a", size = 2.5, alpha = 0.9) +
    ggplot2::scale_x_discrete(expand = ggplot2::expansion(add = 0.5)) +
    ggplot2::coord_flip() +
    ggplot2::labs(x = "", y = "Frequency") +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(color = "#E0E0E0", linewidth = 0.3),
      panel.grid.minor = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.line.y = ggplot2::element_blank(),
      axis.ticks.x = ggplot2::element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks.y = ggplot2::element_blank(),
      axis.text.x = ggplot2::element_text(size = 16, color = "#3B3B3B", margin = ggplot2::margin(t = 3)),
      axis.text.y = ggplot2::element_text(size = 16, color = "#3B3B3B", margin = ggplot2::margin(r = 3)),
      axis.title = ggplot2::element_text(size = 16, color = "#0c1f4a"),
      axis.title.x = ggplot2::element_text(margin = ggplot2::margin(t = 5)),
      axis.title.y = ggplot2::element_text(margin = ggplot2::margin(r = 5)),
      plot.margin = ggplot2::margin(t = 5, r = 10, b = 5, l = 5)
    )

  plotly::ggplotly(ggplot_obj, height = height, width = width, tooltip = "text", ...) %>%
    plotly::layout(
      autosize = TRUE,
      margin = list(t = 20, b = 80, l = 80, r = 20),
      xaxis = list(
        title = list(text = "Frequency", font = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = "", font = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      hoverlabel = list(
        bgcolor = "#0c1f4a",
        font = list(size = 16, color = "white", family = "Roboto, sans-serif"),
        bordercolor = "#0c1f4a",
        align = "left"
      )
    )
}


#' Plot N-gram Frequency
#'
#' @description
#' Creates a bar plot showing n-gram frequencies with optional highlighting
#' of selected n-grams. Supports both detected n-grams and selected multi-word expressions.
#'
#' @param ngram_data Data frame containing n-gram data with columns:
#'   \itemize{
#'     \item \code{collocation}: The n-gram text
#'     \item \code{count}: Frequency count
#'     \item \code{lambda}: (optional) Lambda statistic
#'     \item \code{z}: (optional) Z-score statistic
#'   }
#' @param top_n Number of top n-grams to display (default: 30)
#' @param selected Character vector of selected n-grams to highlight (default: NULL)
#' @param title Plot title (default: "N-gram Frequency")
#' @param highlight_color Color for highlighted bars (default: "#10B981")
#' @param default_color Color for non-highlighted bars (default: "#6B7280")
#' @param height Plot height in pixels (default: 500)
#' @param width Plot width in pixels (default: NULL for auto)
#' @param show_stats Whether to show lambda and z-score in hover (default: TRUE)
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
#'
#' @examples
#' if (interactive()) {
#'   ngram_df <- data.frame(
#'     collocation = c("machine learning", "deep learning", "neural network"),
#'     count = c(150, 120, 90),
#'     lambda = c(5.2, 4.8, 4.1),
#'     z = c(12.3, 10.5, 9.2)
#'   )
#'   plot_ngram_frequency(ngram_df, selected = c("machine learning"))
#' }
plot_ngram_frequency <- function(ngram_data,
                                  top_n = 30,
                                  selected = NULL,
                                  title = "N-gram Frequency",
                                  highlight_color = "#10B981",
                                  default_color = "#6B7280",
                                  height = 500,
                                  width = NULL,
                                  show_stats = TRUE) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (is.null(ngram_data) || nrow(ngram_data) == 0) {
    return(plotly::plot_ly(type = "scatter", mode = "markers") %>%
      plotly::layout(
        xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        annotations = list(
          list(
            text = "No n-grams detected. Adjust parameters and click 'Detect N-grams'",
            x = 0.5, y = 0.5,
            xref = "paper", yref = "paper",
            showarrow = FALSE,
            font = list(size = 16, color = "#6B7280", family = "Roboto")
          )
        )
      ))
  }

  top_ngrams <- utils::head(ngram_data, top_n)

  top_ngrams <- top_ngrams %>%
    dplyr::mutate(
      order_rank = dplyr::row_number(),
      collocation_ordered = factor(collocation, levels = rev(collocation))
    )

  is_selected <- if (!is.null(selected)) {
    top_ngrams$collocation %in% selected
  } else {
    rep(FALSE, nrow(top_ngrams))
  }

  hover_text <- if (show_stats && "lambda" %in% names(top_ngrams) && "z" %in% names(top_ngrams)) {
    paste0(
      top_ngrams$collocation, "\n",
      "Frequency: ", top_ngrams$count, "\n",
      "Lambda: ", round(top_ngrams$lambda, 2), "\n",
      "Z-score: ", round(top_ngrams$z, 2)
    )
  } else {
    paste0(
      top_ngrams$collocation, "\n",
      "Frequency: ", top_ngrams$count
    )
  }

  p <- plotly::plot_ly(
    data = top_ngrams,
    x = ~collocation_ordered,
    y = ~count,
    type = "bar",
    marker = list(
      color = ifelse(is_selected, highlight_color, default_color),
      line = list(
        color = ifelse(is_selected, "#337ab7", "#4B5563"),
        width = 1
      )
    ),
    hoverinfo = "text",
    hovertext = hover_text,
    textposition = "none",
    height = height,
    width = width
  )

  p %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = "",
        tickangle = -45,
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = "Frequency",
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      margin = list(b = 150, l = 60, r = 20, t = 60),
      showlegend = FALSE,
      hoverlabel = list(
        align = "left",
        font = list(size = 16, color = "white", family = "Roboto, sans-serif"),
        bgcolor = "#0c1f4a"
      )
    ) %>%
    plotly::config(displayModeBar = TRUE)
}


#' Plot Multi-Word Expression Frequency
#'
#' @description
#' Creates a bar plot showing multi-word expression frequencies with optional
#' source-based coloring to distinguish between detected and manually added expressions.
#'
#' @param mwe_data Data frame containing MWE data with columns:
#'   \itemize{
#'     \item \code{feature}: The multi-word expression text
#'     \item \code{frequency}: Frequency count
#'     \item \code{rank}: (optional) Rank of the expression
#'     \item \code{docfreq}: (optional) Document frequency
#'     \item \code{source}: (optional) Source category (e.g., "Top 20", "Manual")
#'   }
#' @param title Plot title (default: "Multi-Word Expression Frequency")
#' @param color_by_source Whether to color bars by source column (default: TRUE)
#' @param primary_color Color for primary/top expressions (default: "#10B981")
#' @param secondary_color Color for secondary/manual expressions (default: "#A855F7")
#' @param height Plot height in pixels (default: 500)
#' @param width Plot width in pixels (default: NULL for auto)
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
plot_mwe_frequency <- function(mwe_data,
                                title = "Multi-Word Expression Frequency",
                                color_by_source = TRUE,
                                primary_color = "#10B981",
                                secondary_color = "#A855F7",
                                height = 500,
                                width = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (is.null(mwe_data) || nrow(mwe_data) == 0) {
    return(create_empty_plot_message("No multi-word expressions found"))
  }

  if (color_by_source && "source" %in% names(mwe_data)) {
    bar_colors <- ifelse(mwe_data$source == "Top 20", primary_color, secondary_color)
  } else {
    bar_colors <- primary_color
  }

  hover_text <- if (all(c("rank", "docfreq", "source") %in% names(mwe_data))) {
    paste0(
      mwe_data$feature, "\n",
      "Frequency: ", mwe_data$frequency, "\n",
      "Rank: ", mwe_data$rank, "\n",
      "Doc Frequency: ", mwe_data$docfreq, "\n",
      "Source: ", mwe_data$source
    )
  } else {
    paste0(mwe_data$feature, "\nFrequency: ", mwe_data$frequency)
  }

  plotly::plot_ly(
    data = mwe_data,
    x = ~stats::reorder(feature, frequency),
    y = ~frequency,
    type = "bar",
    marker = list(color = bar_colors),
    hoverinfo = "text",
    hovertext = hover_text,
    textposition = "none",
    height = height,
    width = width
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = "",
        tickangle = -45,
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = "Frequency",
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      margin = list(b = 150, l = 60, r = 20, t = 60),
      hoverlabel = list(
        align = "left",
        font = list(size = 16, color = "white", family = "Roboto, sans-serif"),
        bgcolor = "#0c1f4a"
      )
    )
}



################################################################################
# KEYWORD EXTRACTION
################################################################################

#' Extract Keywords Using TF-IDF
#'
#' @description
#' Extracts top keywords from a document-feature matrix using TF-IDF weighting.
#'
#' @param dfm A quanteda dfm object
#' @param top_n Number of top keywords to extract (default: 20)
#' @param normalize Logical, whether to normalize TF-IDF scores to 0-1 range (default: FALSE)
#'
#' @return Data frame with columns: Keyword, TF_IDF_Score, Frequency
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' library(quanteda)
#' corp <- corpus(c("text analysis", "data mining", "text mining"))
#' dfm_obj <- dfm(tokens(corp))
#' keywords <- extract_keywords_tfidf(dfm_obj, top_n = 5)
#' print(keywords)
#' }
extract_keywords_tfidf <- function(dfm,
                                   top_n = 20,
                                   normalize = FALSE) {

  if (!requireNamespace("quanteda", quietly = TRUE)) {
    stop("Package 'quanteda' is required.")
  }

  tfidf <- quanteda::dfm_tfidf(dfm)

  feature_scores <- colSums(as.matrix(tfidf))
  feature_freq <- colSums(as.matrix(dfm))

  if (normalize) {
    max_score <- max(feature_scores)
    if (max_score > 0) {
      feature_scores <- feature_scores / max_score
    }
  }

  top_features <- sort(feature_scores, decreasing = TRUE)[1:min(top_n, length(feature_scores))]

  data.frame(
    Keyword = names(top_features),
    TF_IDF_Score = unname(top_features),
    Frequency = unname(feature_freq[names(top_features)]),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}


#' Extract Keywords Using Statistical Keyness
#'
#' @description
#' Extracts distinctive keywords by comparing document groups using log-likelihood ratio (G-squared).
#'
#' @param dfm A quanteda dfm object
#' @param target Target document indices or logical vector
#' @param top_n Number of top keywords to extract (default: 20)
#' @param measure Keyness measure: "lr" (log-likelihood) or "chi2" (default: "lr")
#'
#' @return Data frame with columns: Keyword, Keyness_Score
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' library(quanteda)
#' corp <- corpus(c("positive text", "negative text", "positive words"))
#' dfm_obj <- dfm(tokens(corp))
#' # Compare first document vs rest
#' keywords <- extract_keywords_keyness(dfm_obj, target = 1)
#' print(keywords)
#' }
extract_keywords_keyness <- function(dfm,
                                     target,
                                     top_n = 20,
                                     measure = "lr") {

  if (!requireNamespace("quanteda.textstats", quietly = TRUE)) {
    stop("Package 'quanteda.textstats' is required.")
  }

  if (quanteda::ndoc(dfm) < 2) {
    return(data.frame(
      Keyword = character(),
      Keyness_Score = numeric(),
      stringsAsFactors = FALSE
    ))
  }

  keyness <- quanteda.textstats::textstat_keyness(
    dfm,
    target = target,
    measure = measure
  )

  keyness_top <- head(keyness[order(-abs(keyness$G2)), ], min(top_n, nrow(keyness)))

  data.frame(
    Keyword = keyness_top$feature,
    Keyness_Score = keyness_top$G2,
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}


#' Plot TF-IDF Keywords
#'
#' @description
#' Creates a horizontal bar plot of top keywords by TF-IDF score.
#'
#' @param tfidf_data Data frame from extract_keywords_tfidf()
#' @param title Plot title (default: "Top Keywords by TF-IDF Score")
#' @param normalized Logical, whether scores are normalized (for label) (default: FALSE)
#'
#' @return A plotly bar chart
#'
#' @family lexical
#' @export
plot_tfidf_keywords <- function(tfidf_data,
                                 title = NULL,
                                 normalized = FALSE) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required.")
  }

  tfidf_data_sorted <- tfidf_data[order(tfidf_data$TF_IDF_Score, decreasing = FALSE), ]

  score_label <- if (normalized) "TF-IDF Score (Normalized)" else "TF-IDF Score"

  if (is.null(title)) {
    title <- paste("Top Keywords by", score_label)
  }

  plotly::plot_ly(
    x = tfidf_data_sorted$TF_IDF_Score,
    y = tfidf_data_sorted$Keyword,
    type = "bar",
    orientation = "h",
    marker = list(color = "#337ab7"),
    text = ~paste0(
      "Keyword: ", tfidf_data_sorted$Keyword, "<br>",
      score_label, ": ", round(tfidf_data_sorted$TF_IDF_Score, 4), "<br>",
      "Frequency: ", tfidf_data_sorted$Frequency
    ),
    textposition = "none",
    hovertemplate = "%{text}<extra></extra>",
    hoverlabel = get_plotly_hover_config("#E3F2FD", "#1976D2")
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = list(text = score_label),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = ""),
        categoryorder = "trace",
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      margin = list(l = 150, r = 20, t = 60, b = 60),
      font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
      hoverlabel = list(
        font = list(size = 16, family = "Roboto, sans-serif"),
        align = "left"
      )
    ) %>%
    plotly::config(displayModeBar = TRUE)
}


#' Plot Statistical Keyness
#'
#' @description
#' Creates a horizontal bar plot of distinctive keywords by keyness score.
#'
#' @param keyness_data Data frame from extract_keywords_keyness()
#' @param title Plot title (default: "Top Keywords by Keyness (G-squared)")
#' @param group_label Optional label for the target group (default: NULL)
#'
#' @return A plotly bar chart
#'
#' @family lexical
#' @export
plot_keyness_keywords <- function(keyness_data,
                                  title = NULL,
                                  group_label = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required.")
  }

  if (nrow(keyness_data) == 0) {
    return(plot_error("Keyness analysis requires multiple documents"))
  }

  keyness_data_sorted <- keyness_data[order(abs(keyness_data$Keyness_Score), decreasing = FALSE), ]

  if (is.null(title)) {
    title <- if (!is.null(group_label)) {
      paste0("Top Keywords by Keyness (G\u00b2) - Grouped by ", group_label)
    } else {
      "Top Keywords by Keyness (G\u00b2)"
    }
  }

  plotly::plot_ly(
    x = keyness_data_sorted$Keyness_Score,
    y = keyness_data_sorted$Keyword,
    type = "bar",
    orientation = "h",
    marker = list(color = "#337ab7"),
    text = ~paste0(
      "Keyword: ", keyness_data_sorted$Keyword, "<br>",
      "Keyness Score (G\u00b2): ", round(keyness_data_sorted$Keyness_Score, 2)
    ),
    textposition = "none",
    hovertemplate = "%{text}<extra></extra>",
    hoverlabel = get_plotly_hover_config("#E3F2FD", "#1976D2")
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = list(text = "Keyness Score (G\u00b2)"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = ""),
        categoryorder = "trace",
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      margin = list(l = 150, r = 20, t = 60, b = 60),
      font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
      hoverlabel = list(
        font = list(size = 16, family = "Roboto, sans-serif"),
        align = "left"
      )
    ) %>%
    plotly::config(displayModeBar = TRUE)
}


#' Plot Keyword Comparison (TF-IDF vs Frequency)
#'
#' @description
#' Creates a grouped bar plot comparing TF-IDF scores with term frequencies.
#'
#' @param tfidf_data Data frame from extract_keywords_tfidf()
#' @param top_n Number of keywords to display (default: 10)
#' @param title Plot title (default: auto-generated)
#' @param normalized Logical, whether TF-IDF scores are normalized (default: FALSE)
#'
#' @return A plotly grouped bar chart
#'
#' @family lexical
#' @export
plot_keyword_comparison <- function(tfidf_data,
                                    top_n = 10,
                                    title = NULL,
                                    normalized = FALSE) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required.")
  }

  top_keywords <- head(tfidf_data, top_n)

  score_label <- if (normalized) "TF-IDF Score (Normalized)" else "TF-IDF Score"

  if (is.null(title)) {
    title <- paste0("Top Keywords: ", score_label, " vs Frequency")
  }

  plotly::plot_ly(
    data = top_keywords,
    x = ~Keyword,
    y = ~TF_IDF_Score,
    type = "bar",
    name = "TF-IDF",
    marker = list(color = "#337ab7"),
    hovertemplate = paste0("%{x}<br>TF-IDF: %{y:.4f}<extra></extra>"),
    textposition = "none",
    hoverlabel = get_plotly_hover_config("#E3F2FD", "#1976D2")
  ) %>%
    plotly::add_trace(
      y = ~Frequency / max(Frequency) * max(TF_IDF_Score),
      name = "Frequency",
      marker = list(color = "#5cb85c"),
      hovertemplate = "%{x}<br>Frequency: %{text}<extra></extra>",
      text = ~Frequency,
      textposition = "none",
      hoverlabel = get_plotly_hover_config("#E8F5E9", "#2E7D32")
    ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = list(text = "Keywords"),
        tickangle = -45,
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = "Score"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      barmode = "group",
      margin = list(l = 60, r = 100, t = 60, b = 120),
      font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
      hoverlabel = list(align = "left", font = list(size = 16)),
      showlegend = TRUE,
      legend = list(
        font = list(size = 16, family = "Roboto, sans-serif"),
        orientation = "v",
        x = 1.02,
        y = 0.5,
        xanchor = "left",
        yanchor = "middle"
      )
    ) %>%
    plotly::config(displayModeBar = TRUE)
}



################################################################################
# READABILITY
################################################################################

# Readability Analysis Functions
#
# Functions for calculating and visualizing text readability metrics.

#' Plot Readability Distribution
#'
#' @description
#' Creates a boxplot showing the overall distribution of a readability metric.
#'
#' @param readability_data Data frame from calculate_text_readability()
#' @param metric Metric to plot (e.g., "flesch", "flesch_kincaid", "gunning_fog")
#' @param title Plot title (default: auto-generated)
#'
#' @return A plotly boxplot
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' texts <- c("Simple text.", "More complex sentence structure here.")
#' readability <- calculate_text_readability(texts)
#' plot <- plot_readability_distribution(readability, "flesch")
#' print(plot)
#' }
plot_readability_distribution <- function(readability_data,
                                          metric,
                                          title = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required.")
  }

  if (!metric %in% names(readability_data)) {
    stop(paste("Metric", metric, "not found in readability data"))
  }

  if (is.null(title)) {
    title <- paste(metric, "- Overall Distribution")
  }

  metric_values <- readability_data[[metric]]
  metric_values <- metric_values[is.finite(metric_values)]

  if (length(metric_values) == 0) {
    return(plot_error("No valid data for selected metric"))
  }

  plotly::plot_ly(
    y = metric_values,
    type = "box",
    name = "Distribution",
    marker = list(color = "#4A90E2"),
    line = list(color = "#0c1f4a"),
    fillcolor = "rgba(74, 144, 226, 0.7)",
    hoverinfo = "y",
    hoverlabel = get_plotly_hover_config()
  ) %>%
    apply_standard_plotly_layout(
      title = title,
      yaxis_title = metric,
      margin = list(t = 60, b = 60, l = 80, r = 40)
    )

}


#' Plot Readability by Group
#'
#' @description
#' Creates grouped boxplots comparing readability across categories.
#'
#' @param readability_data Data frame from calculate_text_readability()
#' @param metric Metric to plot
#' @param group_var Name of grouping variable column
#' @param title Plot title (default: auto-generated)
#'
#' @return A plotly boxplot
#'
#' @family lexical
#' @export
plot_readability_by_group <- function(readability_data,
                                      metric,
                                      group_var,
                                      title = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE) || !requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Packages 'plotly' and 'ggplot2' are required.")
  }

  if (!metric %in% names(readability_data)) {
    stop(paste("Metric", metric, "not found in readability data"))
  }

  if (!group_var %in% names(readability_data)) {
    stop(paste("Group variable", group_var, "not found in data"))
  }

  if (is.null(title)) {
    title <- paste(metric, "by", group_var)
  }

  plot_data <- readability_data[, c(metric, group_var)]
  names(plot_data) <- c("metric_value", "group")

  plot_data$metric_value <- round(plot_data$metric_value, 2)
  plot_data <- plot_data[is.finite(plot_data$metric_value), ]

  colors <- c("#4A90E2", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6",
              "#1ABC9C", "#E67E22", "#3498DB")

  unique_groups <- unique(plot_data$group)

  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = group, y = metric_value, fill = group)) +
    ggplot2::geom_boxplot(alpha = 0.7) +
    ggplot2::scale_fill_manual(values = colors[1:length(unique_groups)]) +
    ggplot2::labs(x = group_var, y = metric) +
    create_standard_ggplot_theme() +
    ggplot2::theme(legend.position = "none") +
    ggplot2::ggtitle(title)

  plotly::ggplotly(p, tooltip = "y") %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      hoverlabel = get_plotly_hover_config(),
      margin = list(t = 60, b = 100, l = 80, r = 40)
    )
}


#' Plot Top Documents by Readability
#'
#' @description
#' Creates a bar plot of documents ranked by readability metric.
#'
#' @param readability_data Data frame from calculate_text_readability()
#' @param metric Metric to plot
#' @param top_n Number of documents to show (default: 15)
#' @param order Direction: "highest" or "lowest" (default: "highest")
#' @param title Plot title (default: auto-generated)
#'
#' @return A plotly bar chart
#'
#' @family lexical
#' @export
plot_top_readability_documents <- function(readability_data,
                                           metric,
                                           top_n = 15,
                                           order = "highest",
                                           title = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE) || !requireNamespace("dplyr", quietly = TRUE)) {
    stop("Packages 'plotly' and 'dplyr' are required.")
  }

  if (!metric %in% names(readability_data)) {
    stop(paste("Metric", metric, "not found in readability data"))
  }

  # Handle both "Document" and "document" column names
  doc_col <- if ("Document" %in% names(readability_data)) "Document" else "document"

  sorted_data <- readability_data %>%
    dplyr::arrange(if (order == "highest") dplyr::desc(.data[[metric]]) else .data[[metric]]) %>%
    head(top_n)

  if (is.null(title)) {
    title <- paste("Top", top_n, "Documents by", metric)
  }

  sorted_data$tooltip_text <- paste0(
    "<b>", sorted_data[[doc_col]], "</b><br>",
    metric, ": ", round(sorted_data[[metric]], 2)
  )

  text_angle <- if (top_n <= 10) 0 else -45

  plotly::plot_ly(
    data = sorted_data,
    x = as.formula(paste0("~`", doc_col, "`")),
    y = ~.data[[metric]],
    type = "bar",
    marker = list(color = "#4A90E2"),
    text = ~tooltip_text,
    hovertemplate = "%{text}<extra></extra>",
    textposition = "none",
    showlegend = FALSE
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = list(text = "Document"),
        tickangle = text_angle,
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = metric),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      margin = list(t = 60, b = 100, l = 80, r = 40),
      hoverlabel = get_plotly_hover_config()
    ) %>%
    plotly::config(displayModeBar = TRUE)
}

#' Calculate Text Readability
#'
#' @description
#' Calculates multiple readability metrics for texts including Flesch Reading Ease,
#' Flesch-Kincaid Grade Level, Gunning FOG index, and others. Optionally includes
#' lexical diversity metrics and sentence statistics.
#'
#' @param texts Character vector of texts to analyze
#' @param metrics Character vector of readability metrics to calculate.
#'   Options: "flesch", "flesch_kincaid", "gunning_fog", "smog", "ari", "coleman_liau"
#' @param include_lexical_diversity Logical, include TTR and MTLD (default: TRUE)
#' @param include_sentence_stats Logical, include average sentence length (default: TRUE)
#' @param dfm_for_lexdiv Optional pre-computed DFM for lexical diversity calculation
#' @param doc_names Optional character vector of document names
#'
#' @return A data frame with document names and readability scores
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' texts <- c(
#'   "This is simple text.",
#'   "This sentence contains more complex vocabulary and structure."
#' )
#' readability <- calculate_text_readability(texts)
#' print(readability)
#' }
#'
#' @importFrom quanteda corpus tokens dfm docnames
#' @importFrom quanteda.textstats textstat_readability textstat_lexdiv
calculate_text_readability <- function(texts,
                                      metrics = c("flesch", "flesch_kincaid", "gunning_fog"),
                                      include_lexical_diversity = TRUE,
                                      include_sentence_stats = TRUE,
                                      dfm_for_lexdiv = NULL,
                                      doc_names = NULL) {

  if (!requireNamespace("quanteda.textstats", quietly = TRUE)) {
    stop("Package 'quanteda.textstats' is required. Please install it.")
  }

  if (is.null(doc_names)) {
    doc_names <- paste0("Doc ", seq_along(texts))
  }

  corp <- quanteda::corpus(texts)
  quanteda::docnames(corp) <- doc_names

  measure_map <- c(
    "flesch" = "Flesch",
    "flesch_kincaid" = "Flesch.Kincaid",
    "gunning_fog" = "FOG",
    "smog" = "SMOG",
    "ari" = "ARI",
    "coleman_liau" = "Coleman.Liau.short"
  )

  valid_metrics <- intersect(metrics, names(measure_map))
  if (length(valid_metrics) == 0) {
    stop("No valid metrics specified. Available metrics: ",
         paste(names(measure_map), collapse = ", "))
  }

  mapped_metrics <- measure_map[valid_metrics]
  names(mapped_metrics) <- valid_metrics

  # Batch all readability metrics in single call for performance
  all_scores <- list()
  tryCatch({
    # Call textstat_readability once with all measures
    batch_scores <- quanteda.textstats::textstat_readability(corp, measure = unname(mapped_metrics))

    # Extract scores for each metric
    for (i in seq_along(valid_metrics)) {
      metric <- valid_metrics[i]
      measure_name <- mapped_metrics[i]
      if (measure_name %in% names(batch_scores)) {
        all_scores[[metric]] <- batch_scores[[measure_name]]
      } else {
        all_scores[[metric]] <- rep(NA, length(texts))
      }
    }
  }, error = function(e) {
    # Fallback to individual calls if batch fails
    warning("Batch readability calculation failed, falling back to individual metrics: ", e$message)
    for (i in seq_along(valid_metrics)) {
      metric <- valid_metrics[i]
      measure_name <- mapped_metrics[i]
      tryCatch({
        score <- quanteda.textstats::textstat_readability(corp, measure = measure_name)
        all_scores[[metric]] <<- score[[2]]
      }, error = function(e2) {
        warning(paste("Could not calculate", metric, ":", e2$message))
        all_scores[[metric]] <<- rep(NA, length(texts))
      })
    }
  })

  readability_scores <- data.frame(Document = doc_names, stringsAsFactors = FALSE)
  for (metric in names(all_scores)) {
    readability_scores[[metric]] <- all_scores[[metric]]
  }

  if (include_lexical_diversity) {
    if (is.null(dfm_for_lexdiv)) {
      toks <- quanteda::tokens(texts, remove_punct = TRUE)
      dfm_for_lexdiv <- quanteda::dfm(toks)
    }

    tryCatch({
      lexical_diversity <- quanteda.textstats::textstat_lexdiv(
        dfm_for_lexdiv,
        measure = c("TTR", "MTLD")
      )

      readability_scores$`Lexical Diversity (TTR)` <- NA
      readability_scores$`Lexical Diversity (MTLD)` <- NA

      if (nrow(lexical_diversity) == nrow(readability_scores)) {
        readability_scores$`Lexical Diversity (TTR)` <- lexical_diversity$TTR
        readability_scores$`Lexical Diversity (MTLD)` <- lexical_diversity$MTLD
      } else {
        warning(paste0("Could not calculate lexical diversity: row mismatch (",
                       nrow(lexical_diversity), " vs ", nrow(readability_scores), ")"))
      }
    }, error = function(e) {
      warning("Could not calculate lexical diversity: ", e$message)
      readability_scores$`Lexical Diversity (TTR)` <<- NA
      readability_scores$`Lexical Diversity (MTLD)` <<- NA
    })
  }

  if (include_sentence_stats) {
    avg_sentence_length <- sapply(texts, function(t) {
      sents <- unlist(strsplit(t, "[.!?]+"))
      sents <- sents[nzchar(trimws(sents))]
      words <- unlist(strsplit(paste(sents, collapse = " "), "\\s+"))
      if (length(sents) == 0) return(NA)
      length(words) / length(sents)
    })
    readability_scores$`Avg Sentence Length` <- avg_sentence_length
  }

  return(readability_scores)
}




#' Plot Term Frequency Trends by Continuous Variable
#'
#' @description
#' Creates a faceted line plot showing how term frequencies vary across
#' a continuous variable (e.g., year, time period).
#'
#' @param term_data Data frame containing term frequencies with columns:
#'   continuous_var, term, and word_frequency
#' @param continuous_var Name of the continuous variable column
#' @param terms Character vector of terms to display (optional, filters if provided)
#' @param title Plot title (default: NULL, auto-generated)
#' @param height Plot height in pixels (default: 600)
#' @param width Plot width in pixels (default: NULL, auto)
#'
#' @return A plotly object with faceted line plots
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' term_df <- data.frame(
#'   year = rep(2010:2020, each = 3),
#'   term = rep(c("learning", "education", "technology"), 11),
#'   word_frequency = sample(10:100, 33, replace = TRUE)
#' )
#' plot_term_trends_continuous(term_df, "year", c("learning", "education"))
#' }
plot_term_trends_continuous <- function(term_data,
                                         continuous_var,
                                         terms = NULL,
                                         title = NULL,
                                         height = 600,
                                         width = NULL) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required. Please install it.")
  }
  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }
  if (!requireNamespace("scales", quietly = TRUE)) {
    stop("Package 'scales' is required. Please install it.")
  }

  if (!continuous_var %in% names(term_data)) {
    stop("Continuous variable '", continuous_var, "' not found in data")
  }

  if (!"term" %in% names(term_data) && !"word" %in% names(term_data)) {
    stop("term or word column not found in data")
  }

  if ("word" %in% names(term_data) && !"term" %in% names(term_data)) {
    term_data$term <- term_data$word
  }

  if (!"word_frequency" %in% names(term_data) && !"count" %in% names(term_data)) {
    stop("word_frequency or count column not found in data")
  }

  if ("count" %in% names(term_data) && !"word_frequency" %in% names(term_data)) {
    term_data$word_frequency <- term_data$count
  }

  if (!is.null(terms)) {
    term_data <- term_data %>%
      dplyr::filter(term %in% terms) %>%
      dplyr::mutate(term = factor(term, levels = terms))
  }

  if (is.null(title)) {
    title <- paste("Term Frequency by", continuous_var)
  }

  p <- ggplot2::ggplot(
    term_data,
    ggplot2::aes(
      x = .data[[continuous_var]],
      y = word_frequency,
      group = term
    )
  ) +
    ggplot2::geom_point(color = "#337ab7", alpha = 0.6, size = 2.5) +
    ggplot2::geom_line(color = "#337ab7", alpha = 0.6, linewidth = 0.5) +
    ggplot2::facet_wrap(~term, scales = "free") +
    ggplot2::scale_y_continuous(labels = scales::number_format(accuracy = 1)) +
    ggplot2::labs(y = "Word Frequency", x = continuous_var) +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      legend.position = "none",
      axis.line = ggplot2::element_line(color = "#3B3B3B", linewidth = 0.3),
      axis.ticks = ggplot2::element_line(color = "#3B3B3B", linewidth = 0.3),
      strip.text.x = ggplot2::element_text(size = 16, color = "#0c1f4a", family = "Roboto"),
      axis.text.x = ggplot2::element_text(size = 16, color = "#3B3B3B", family = "Roboto"),
      axis.text.y = ggplot2::element_text(size = 16, color = "#3B3B3B", family = "Roboto"),
      axis.title = ggplot2::element_text(size = 16, color = "#0c1f4a", family = "Roboto"),
      axis.title.x = ggplot2::element_text(margin = ggplot2::margin(t = 15)),
      axis.title.y = ggplot2::element_text(margin = ggplot2::margin(r = 15)),
      plot.margin = ggplot2::margin(t = 5, r = 10, b = 25, l = 15)
    )

  plot_args <- list(p)
  if (!is.null(height)) plot_args$height <- height
  if (!is.null(width)) plot_args$width <- width

  p_plot <- do.call(plotly::ggplotly, plot_args)

  for (i in seq_along(p_plot$x$layout$annotations)) {
    p_plot$x$layout$annotations[[i]]$font <- list(
      size = 16,
      color = "#0c1f4a",
      family = "Roboto, sans-serif"
    )
  }

  axis_names <- names(p_plot$x$layout)
  for (axis_name in axis_names) {
    if (grepl("^xaxis", axis_name)) {
      p_plot$x$layout[[axis_name]]$tickfont <- list(
        size = 14,
        color = "#3B3B3B",
        family = "Roboto, sans-serif"
      )
      p_plot$x$layout[[axis_name]]$titlefont <- list(
        size = 16,
        color = "#0c1f4a",
        family = "Roboto, sans-serif"
      )
    }
    if (grepl("^yaxis", axis_name)) {
      p_plot$x$layout[[axis_name]]$tickfont <- list(
        size = 14,
        color = "#3B3B3B",
        family = "Roboto, sans-serif"
      )
      p_plot$x$layout[[axis_name]]$titlefont <- list(
        size = 16,
        color = "#0c1f4a",
        family = "Roboto, sans-serif"
      )
    }
  }

  p_plot %>%
    plotly::layout(
      margin = list(l = 80, r = 150, t = 40, b = 100),
      font = list(
        family = "Roboto, sans-serif",
        size = 14,
        color = "#3B3B3B"
      ),
      hoverlabel = list(
        font = list(size = 14, family = "Roboto, sans-serif")
      )
    ) %>%
    plotly::config(displayModeBar = TRUE)
}



#' Plot Part-of-Speech Tag Frequencies
#'
#' @description
#' Creates a bar plot showing the frequency distribution of part-of-speech tags.
#'
#' @param pos_data Data frame containing POS data with columns:
#'   \itemize{
#'     \item \code{pos}: Part-of-speech tag
#'     \item \code{n}: (optional) Pre-computed frequency count
#'   }
#'   If \code{n} is not present, frequencies will be computed from the data.
#' @param top_n Number of top POS tags to display (default: 20)
#' @param title Plot title (default: "Part-of-Speech Tag Frequency")
#' @param color Bar color (default: "#337ab7")
#' @param height Plot height in pixels (default: 500)
#' @param width Plot width in pixels (default: NULL for auto)
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
#'
#' @examples
#' if (interactive()) {
#'   pos_df <- data.frame(
#'     pos = c("NOUN", "VERB", "ADJ", "ADV", "PRON"),
#'     n = c(500, 400, 250, 150, 100)
#'   )
#'   plot_pos_frequencies(pos_df)
#' }
plot_pos_frequencies <- function(pos_data,
                                  top_n = 20,
                                  title = "Part-of-Speech Tag Frequency",
                                  color = "#337ab7",
                                  height = 500,
                                  width = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (is.null(pos_data) || nrow(pos_data) == 0) {
    return(plotly::plot_ly() %>%
      plotly::layout(
        xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        annotations = list(
          list(
            text = "No POS data available",
            x = 0.5, y = 0.5,
            xref = "paper", yref = "paper",
            showarrow = FALSE,
            font = list(size = 16, color = "#6B7280", family = "Roboto")
          )
        )
      ))
  }

  if (!"n" %in% names(pos_data)) {
    pos_freq <- pos_data %>%
      dplyr::count(pos, sort = TRUE) %>%
      dplyr::slice_head(n = top_n)
  } else {
    pos_freq <- pos_data %>%
      dplyr::arrange(dplyr::desc(n)) %>%
      dplyr::slice_head(n = top_n)
  }

  plotly::plot_ly(
    data = pos_freq,
    x = ~stats::reorder(pos, n),
    y = ~n,
    type = "bar",
    marker = list(color = color),
    hoverinfo = "text",
    hovertext = ~paste0(pos, "\nFrequency: ", n),
    height = height,
    width = width
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = "POS Tag",
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = "Frequency",
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      margin = list(b = 100, l = 60, r = 20, t = 60),
      hoverlabel = list(
        align = "left",
        font = list(size = 14, color = "white", family = "Roboto, sans-serif"),
        bgcolor = "#0c1f4a"
      )
    )
}


#' Plot Named Entity Frequencies
#'
#' @description
#' Creates a bar plot showing the frequency distribution of named entity types.
#'
#' @param entity_data Data frame containing entity data with columns:
#'   \itemize{
#'     \item \code{entity}: Named entity type (e.g., "PERSON", "ORG", "GPE")
#'     \item \code{n}: (optional) Pre-computed frequency count
#'   }
#'   If \code{n} is not present, frequencies will be computed from the data.
#' @param top_n Number of top entity types to display (default: 20)
#' @param title Plot title (default: "Named Entity Type Frequency")
#' @param color Bar color (default: "#10B981")
#' @param height Plot height in pixels (default: 500)
#' @param width Plot width in pixels (default: NULL for auto)
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
#'
#' @examples
#' if (interactive()) {
#'   entity_df <- data.frame(
#'     entity = c("PERSON", "ORG", "GPE", "DATE", "MONEY"),
#'     n = c(300, 250, 200, 150, 100)
#'   )
#'   plot_entity_frequencies(entity_df)
#' }
plot_entity_frequencies <- function(entity_data,
                                     top_n = 20,
                                     title = "Named Entity Type Frequency",
                                     color = "#10B981",
                                     height = 500,
                                     width = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (is.null(entity_data) || nrow(entity_data) == 0) {
    return(plotly::plot_ly(type = "scatter", mode = "markers") %>%
      plotly::layout(
        xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
        annotations = list(
          list(
            text = "No named entities found",
            x = 0.5, y = 0.5,
            xref = "paper", yref = "paper",
            showarrow = FALSE,
            font = list(size = 16, color = "#6B7280", family = "Roboto")
          )
        )
      ))
  }

  if (!"n" %in% names(entity_data)) {
    entity_freq <- entity_data %>%
      dplyr::count(entity, sort = TRUE) %>%
      dplyr::slice_head(n = top_n)
  } else {
    entity_freq <- entity_data %>%
      dplyr::arrange(dplyr::desc(n)) %>%
      dplyr::slice_head(n = top_n)
  }

  plotly::plot_ly(
    data = entity_freq,
    x = ~stats::reorder(entity, n),
    y = ~n,
    type = "bar",
    marker = list(color = color),
    hoverinfo = "text",
    hovertext = ~paste0(entity, "\nFrequency: ", n),
    height = height,
    width = width
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = "",
        tickangle = -45,
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = "Frequency",
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      margin = list(b = 150, l = 60, r = 20, t = 60),
      hoverlabel = list(
        align = "left",
        font = list(size = 16, color = "white", family = "Roboto, sans-serif"),
        bgcolor = "#0c1f4a"
      )
    )
}

#' Render displaCy Entity Visualization
#'
#' @description
#' Renders spaCy's displaCy entity visualization as HTML.
#' Highlights named entities with colored labels.
#'
#' @param text Character string to visualize.
#' @param model spaCy model name (default: "en_core_web_sm").
#'
#' @return HTML string with entity highlighting.
#'
#' @family lexical
#' @export
render_displacy_ent <- function(text, model = "en_core_web_sm") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for displaCy visualization.")
  }

  tryCatch({
    spacy <- reticulate::import("spacy")
    displacy_module <- reticulate::import("spacy.displacy")

    # Load model
    nlp <- spacy$load(model)
    doc <- nlp(text)

    # Render as HTML
    html <- displacy_module$render(doc, style = "ent", page = FALSE)

    return(as.character(html))
  }, error = function(e) {
    stop("displaCy rendering failed: ", e$message)
  })
}


#' Render displaCy Dependency Visualization
#'
#' @description
#' Renders spaCy's displaCy dependency visualization as SVG.
#' Shows syntactic structure with arrows between words.
#'
#' @param text Character string to visualize.
#' @param compact Logical; use compact mode for space (default: TRUE).
#' @param model spaCy model name (default: "en_core_web_sm").
#'
#' @return SVG string with dependency tree.
#'
#' @family lexical
#' @export
render_displacy_dep <- function(text, compact = TRUE, model = "en_core_web_sm") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for displaCy visualization.")
  }

  tryCatch({
    spacy <- reticulate::import("spacy")
    displacy_module <- reticulate::import("spacy.displacy")

    # Load model
    nlp <- spacy$load(model)
    doc <- nlp(text)

    # Render options
    options <- list(compact = compact)

    # Render as SVG
    svg <- displacy_module$render(doc, style = "dep", options = options, page = FALSE)

    return(as.character(svg))
  }, error = function(e) {
    stop("displaCy rendering failed: ", e$message)
  })
}
