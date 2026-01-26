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
#' @importFrom Matrix colSums
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
#' This function requires the Python
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

  parsed <- spacy_parse_full(
    tokens,
    pos = TRUE,
    tag = TRUE,
    lemma = include_lemma,
    entity = include_entity,
    dependency = include_dependency,
    model = model
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

  # Use spacy_parse_full with morphology enabled
  parsed <- spacy_parse_full(
    tokens,
    pos = include_pos,
    tag = include_pos,
    lemma = include_lemma,
    entity = FALSE,
    dependency = FALSE,
    morph = TRUE,
    model = model
  )

  # Parse the morph string into individual feature columns
  if ("morph" %in% names(parsed) && nrow(parsed) > 0) {
    parsed <- parse_morphology_string(parsed, features)
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
      plotly::plot_ly(type = "scatter", mode = "markers") %>%
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
      plotly::plot_ly(type = "scatter", mode = "markers") %>%
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
    vapply(freq_df$Value, function(v) {
      if (v %in% names(colors)) colors[[v]] else "#6B7280"
    }, character(1))
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
      title = list(
        text = title,
        font = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = "",
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = "Frequency",
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      margin = list(t = 50, b = 50, l = 60, r = 20),
      hoverlabel = list(
        bgcolor = "#0c1f4a",
        font = list(size = 16, color = "white", family = "Roboto, sans-serif"),
        bordercolor = "#0c1f4a",
        align = "left"
      )
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
#' This function requires the Python
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

  # Use spacy_parse_full with entity enabled
  parsed <- spacy_parse_full(
    tokens,
    pos = include_pos,
    tag = include_pos,
    lemma = include_lemma,
    entity = TRUE,
    dependency = FALSE,
    model = model
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

  # For MTLD with DFM input, proper token sequences are needed
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
        mtld_values <- vapply(seq_len(quanteda::ndoc(tokens_for_mtld)), function(i) {
          doc_tokens <- as.character(tokens_for_mtld[[i]])
          if (length(doc_tokens) < 10) return(NA_real_)
          calculate_mtld(doc_tokens)
        }, numeric(1))

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
      avg_sentence_length <- vapply(texts, function(t) {
        sents <- unlist(strsplit(t, "[.!?]+"))
        sents <- sents[nzchar(trimws(sents))]
        words <- unlist(strsplit(paste(sents, collapse = " "), "\\s+"))
        if (length(sents) == 0) return(NA_real_)
        length(words) / length(sents)
      }, numeric(1))
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
#'   data(SpecialEduTech, package = "TextAnalysisR")
#'   texts <- SpecialEduTech$abstract[1:10]
#'   dfm <- quanteda::dfm(quanteda::tokens(texts))
#'   plot <- plot_word_frequency(dfm, n = 10)
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
#' data(SpecialEduTech, package = "TextAnalysisR")
#' texts <- SpecialEduTech$abstract[1:20]
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
#' data(SpecialEduTech, package = "TextAnalysisR")
#' texts <- SpecialEduTech$abstract[1:10]
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
    avg_sentence_length <- vapply(texts, function(t) {
      sents <- unlist(strsplit(t, "[.!?]+"))
      sents <- sents[nzchar(trimws(sents))]
      words <- unlist(strsplit(paste(sents, collapse = " "), "\\s+"))
      if (length(sents) == 0) return(NA_real_)
      length(words) / length(sents)
    }, numeric(1))
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
    return(plotly::plot_ly(type = "scatter", mode = "markers") %>%
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
#' @param custom_colors Named vector of custom entity type colors (e.g.,
#'   c(CONCEPT = "#00acc1", THEME = "#7c4dff")). Custom colors override defaults.
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
#'
#'   # With custom colors
#'   plot_entity_frequencies(entity_df, custom_colors = c(PERSON = "#ff0000"))
#' }
plot_entity_frequencies <- function(entity_data,
                                     top_n = 20,
                                     title = "Named Entity Type Frequency",
                                     color = NULL,
                                     height = 500,
                                     width = NULL,
                                     custom_colors = NULL) {

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

  entity_colors <- c(
    "PERSON" = "#e91e63", "ORG" = "#1565c0", "GPE" = "#2e7d32",
    "DATE" = "#ef6c00", "MONEY" = "#6a1b9a", "CARDINAL" = "#546e7a",
    "ORDINAL" = "#5d4037", "PERCENT" = "#00838f", "PRODUCT" = "#283593",
    "EVENT" = "#c62828", "WORK_OF_ART" = "#4527a0", "LAW" = "#00695c",
    "LANGUAGE" = "#558b2f", "LOC" = "#0277bd", "FAC" = "#9e9d24",
    "NORP" = "#ff8f00", "TIME" = "#d84315", "QUANTITY" = "#78909c",
    "DISABILITY" = "#ad1457", "PROGRAM" = "#1976d2", "TEST" = "#7b1fa2",
    "CONCEPT" = "#00897b", "TOOL" = "#6d4c41", "METHOD" = "#c2185b",
    "THEME" = "#7c4dff", "CODE" = "#37474f", "CATEGORY" = "#26a69a",
    "CUSTOM" = "#d81b60"
  )

  if (!is.null(custom_colors) && length(custom_colors) > 0) {
    entity_colors[names(custom_colors)] <- custom_colors
  }

  bar_colors <- sapply(entity_freq$entity, function(e) {
    if (e %in% names(entity_colors)) {
      entity_colors[[e]]
    } else {
      "#757575"
    }
  })

  plotly::plot_ly(
    data = entity_freq,
    x = ~stats::reorder(entity, n),
    y = ~n,
    type = "bar",
    marker = list(color = bar_colors),
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
#' @param colors Named list of entity type to color mappings (e.g.,
#'   list(PERSON = "#e91e63", ORG = "#2196f3")). If NULL, uses spaCy defaults.
#'
#' @return HTML string with entity highlighting.
#'
#' @family lexical
#' @export
render_displacy_ent <- function(text, model = "en_core_web_sm", colors = NULL) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for displaCy visualization.")
  }

  tryCatch({
    spacy <- reticulate::import("spacy")
    displacy_module <- reticulate::import("spacy.displacy")

    # Load model
    nlp <- spacy$load(model)
    doc <- nlp(text)

    # Build options with custom colors if provided
    options <- list()
    if (!is.null(colors) && length(colors) > 0) {
      options$colors <- colors
    }

    # Render as HTML
    if (length(options) > 0) {
      html <- displacy_module$render(doc, style = "ent", page = FALSE, options = options)
    } else {
      html <- displacy_module$render(doc, style = "ent", page = FALSE)
    }

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


# =============================================================================
# spaCy NLP Interface Functions
# =============================================================================
# R wrapper functions for spaCy NLP via reticulate.
# Provides direct Python spaCy access for full control
# over all spaCy features including morphology.

# Package-level spaCy instance
.spacy_env <- new.env(parent = emptyenv())

#' Initialize spaCy NLP
#'
#' @description
#' Initialize the spaCy NLP pipeline with the specified model.
#' Uses a cached instance for efficiency.
#'
#' @param model Character; spaCy model name (default: "en_core_web_sm").
#' @param force Logical; force reinitialization even if already initialized.
#'
#' @return Invisibly returns the SpacyNLP Python object.
#'
#' @details
#' Available models:
#' \itemize{
#'   \item \code{en_core_web_sm}: Small English model (fast, no word vectors)
#'   \item \code{en_core_web_md}: Medium English model (word vectors)
#'   \item \code{en_core_web_lg}: Large English model (best accuracy)
#' }
#'
#' @family lexical
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
  if (!force && !is.null(.spacy_env$nlp) && .spacy_env$model == model) {
    return(invisible(.spacy_env$nlp))
  }

  # Import the Python module
  python_path <- system.file("python", package = "TextAnalysisR")
  if (python_path == "") {
    stop("Cannot find Python module directory in TextAnalysisR package")
  }

  tryCatch({
    spacy_module <- reticulate::import_from_path("spacy_nlp", path = python_path)
    nlp <- spacy_module$SpacyNLP(model)
    .spacy_env$nlp <- nlp
    .spacy_env$model <- model
    .spacy_env$module <- spacy_module
    message("spaCy initialized with model: ", model)
    invisible(nlp)
  }, error = function(e) {
    stop("Failed to initialize spaCy: ", e$message,
         "\nMake sure spaCy is installed: pip install spacy",
         "\nAnd download the model: python -m spacy download ", model)
  })
}

#' Check if spaCy is Initialized
#'
#' @description
#' Check whether spaCy has been initialized.
#'
#' @return Logical; TRUE if initialized, FALSE otherwise.
#'
#' @family lexical
#' @export
spacy_initialized <- function() {
  !is.null(.spacy_env$nlp)
}

#' Check if Model Has Word Vectors
#'
#' @description
#' Check if the loaded spaCy model has word vectors for similarity calculations.
#'
#' @return Logical; TRUE if model has vectors, FALSE otherwise.
#'
#' @family lexical
#' @export
spacy_has_vectors <- function() {
  if (!spacy_initialized()) {
    stop("spaCy not initialized. Call init_spacy_nlp() first.")
  }
  .spacy_env$nlp$has_vectors()
}

#' Get spaCy Model Information
#'
#' @description
#' Get information about the currently loaded spaCy model.
#'
#' @return A list with model information including name, language,
#'   pipeline components, and vector availability.
#'
#' @family lexical
#' @export
get_spacy_model_info <- function() {
  if (!spacy_initialized()) {
    stop("spaCy not initialized. Call init_spacy_nlp() first.")
  }
  info <- .spacy_env$nlp$get_model_info()
  as.list(info)
}

#' Parse Texts with spaCy
#'
#' @description
#' Parse texts using spaCy and return token-level annotations.
#' This is the main parsing function for NLP analysis.
#' Works with character vectors or quanteda tokens objects.
#'
#' @param x Character vector of texts OR a quanteda tokens object.
#' @param pos Logical; include coarse POS tags (default: TRUE).
#' @param tag Logical; include fine-grained tags (default: TRUE).
#' @param lemma Logical; include lemmatized forms (default: TRUE).
#' @param entity Logical; include named entity tags (default: FALSE).
#' @param dependency Logical; include dependency relations (default: FALSE).
#' @param morph Logical; include morphological features (default: FALSE).
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data frame with token-level annotations including:
#' \itemize{
#'   \item \code{doc_id}: Document identifier
#'   \item \code{sentence_id}: Sentence number within document
#'   \item \code{token_id}: Token position within sentence
#'   \item \code{token}: Original token text
#'   \item \code{pos}: Coarse POS tag (if pos = TRUE)
#'   \item \code{tag}: Fine-grained tag (if tag = TRUE)
#'   \item \code{lemma}: Lemmatized form (if lemma = TRUE)
#'   \item \code{entity}: Named entity tag (if entity = TRUE)
#'   \item \code{head_token_id}: Head token ID (if dependency = TRUE)
#'   \item \code{dep_rel}: Dependency relation (if dependency = TRUE)
#'   \item \code{morph}: Morphological features string (if morph = TRUE)
#' }
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' # From SpecialEduTech dataset
#' texts <- TextAnalysisR::SpecialEduTech$abstract[1:5]
#' parsed <- spacy_parse_full(texts, morph = TRUE)
#'
#' # From quanteda tokens
#' united <- unite_cols(TextAnalysisR::SpecialEduTech, c("title", "abstract"))
#' tokens <- prep_texts(united, text_field = "united_texts")
#' parsed <- spacy_parse_full(tokens, morph = TRUE)
#' }
spacy_parse_full <- function(x,
                             pos = TRUE,
                             tag = TRUE,
                             lemma = TRUE,
                             entity = FALSE,
                             dependency = FALSE,
                             morph = FALSE,
                             model = "en_core_web_sm") {

  if (!spacy_initialized() || .spacy_env$model != model) {
    init_spacy_nlp(model)
  }

 # Handle quanteda tokens objects
  if (inherits(x, "tokens")) {
    texts <- vapply(as.list(x), function(toks) paste(toks, collapse = " "), character(1))
    doc_names <- quanteda::docnames(x)
  } else if (is.character(x)) {
    texts <- x
    doc_names <- names(x)
    if (is.null(doc_names)) {
      doc_names <- paste0("text", seq_along(texts))
    }
  } else {
    stop("x must be a character vector or quanteda tokens object")
  }

  # Convert to list for Python
  texts_list <- as.list(unname(texts))

  # Call Python method
  result <- .spacy_env$nlp$parse_to_dataframe(
    texts_list,
    include_pos = pos,
    include_tag = tag,
    include_lemma = lemma,
    include_entity = entity,
    include_dependency = dependency,
    include_morph = morph
  )

  # Convert pandas DataFrame to R data.frame
  df <- reticulate::py_to_r(result)

  # Map doc_id from text1, text2, ... to actual document names
  if (nrow(df) > 0 && "doc_id" %in% names(df) && length(doc_names) > 0) {
    doc_id_map <- data.frame(
      old_id = paste0("text", seq_along(doc_names)),
      new_id = doc_names,
      stringsAsFactors = FALSE
    )
    df$doc_id <- doc_id_map$new_id[match(df$doc_id, doc_id_map$old_id)]
  }

  return(df)
}

#' Lemmatize Texts with spaCy
#'
#' @description
#' Perform lemmatization using spaCy with optimized pipeline settings.
#' Disables unnecessary components (NER, parser) for faster processing.
#'
#' @param x Character vector of texts OR a quanteda tokens object.
#' @param batch_size Integer; batch size for processing (default: 100).
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data frame with columns: doc_id, token_id, token, lemma.
#'
#' @details
#' This function disables NER, entity_ruler, and parser components to speed up
#' lemmatization. Use this when you need lemmas without other annotations.
#'
#' @family lexical
#' @export
spacy_lemmatize <- function(x, batch_size = 100, model = "en_core_web_sm") {
  if (!spacy_initialized() || .spacy_env$model != model) {
    init_spacy_nlp(model)
  }

  # Handle quanteda tokens objects
  if (inherits(x, "tokens")) {
    texts <- vapply(as.list(x), function(toks) paste(toks, collapse = " "), character(1))
    doc_names <- quanteda::docnames(x)
  } else if (is.character(x)) {
    texts <- x
    doc_names <- names(x)
    if (is.null(doc_names)) {
      doc_names <- paste0("text", seq_along(texts))
    }
  } else {
    stop("x must be a character vector or quanteda tokens object")
  }

  # Convert to list for Python
  texts_list <- as.list(unname(texts))

  # Call fast Python method
  result <- .spacy_env$nlp$lemmatize(
    texts_list,
    batch_size = as.integer(batch_size)
  )

  # Convert pandas DataFrame to R data.frame
  df <- reticulate::py_to_r(result)

  # Map doc_id from text1, text2, ... to actual document names
  if (nrow(df) > 0 && "doc_id" %in% names(df) && length(doc_names) > 0) {
    doc_id_map <- data.frame(
      old_id = paste0("text", seq_along(doc_names)),
      new_id = doc_names,
      stringsAsFactors = FALSE
    )
    df$doc_id <- doc_id_map$new_id[match(df$doc_id, doc_id_map$old_id)]
  }

  return(df)
}

#' Extract Named Entities with spaCy
#'
#' @description
#' Extract named entities from texts using spaCy NER.
#'
#' @param x Character vector of texts OR a quanteda tokens object.
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data frame with entity information:
#' \itemize{
#'   \item \code{doc_id}: Document identifier
#'   \item \code{text}: Entity text
#'   \item \code{label}: Entity type (PERSON, ORG, GPE, etc.)
#'   \item \code{start_char}: Start character position
#'   \item \code{end_char}: End character position
#' }
#'
#' @family lexical
#' @export
spacy_extract_entities <- function(x, model = "en_core_web_sm") {
  if (!spacy_initialized() || .spacy_env$model != model) {
    init_spacy_nlp(model)
  }

  # Handle quanteda tokens objects
  if (inherits(x, "tokens")) {
    texts <- vapply(as.list(x), function(toks) paste(toks, collapse = " "), character(1))
    doc_names <- quanteda::docnames(x)
  } else if (is.character(x)) {
    texts <- x
    doc_names <- names(x)
    if (is.null(doc_names)) {
      doc_names <- paste0("text", seq_along(texts))
    }
  } else {
    stop("x must be a character vector or quanteda tokens object")
  }

  texts_list <- as.list(unname(texts))
  result <- .spacy_env$nlp$get_entities(texts_list)
  df <- reticulate::py_to_r(result)

  # Map doc_id
  if (nrow(df) > 0 && "doc_id" %in% names(df) && length(doc_names) > 0) {
    doc_id_map <- data.frame(
      old_id = paste0("text", seq_along(doc_names)),
      new_id = doc_names,
      stringsAsFactors = FALSE
    )
    df$doc_id <- doc_id_map$new_id[match(df$doc_id, doc_id_map$old_id)]
  }

  return(df)
}

#' Extract Noun Chunks
#'
#' @description
#' Extract noun chunks (base noun phrases) from texts.
#' Useful for keyphrase extraction.
#'
#' @param x Character vector of texts OR a quanteda tokens object.
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data frame with noun chunk information.
#'
#' @family lexical
#' @export
extract_noun_chunks <- function(x, model = "en_core_web_sm") {
  if (!spacy_initialized() || .spacy_env$model != model) {
    init_spacy_nlp(model)
  }

  if (inherits(x, "tokens")) {
    texts <- vapply(as.list(x), function(toks) paste(toks, collapse = " "), character(1))
    doc_names <- quanteda::docnames(x)
  } else if (is.character(x)) {
    texts <- x
    doc_names <- names(x)
    if (is.null(doc_names)) {
      doc_names <- paste0("text", seq_along(texts))
    }
  } else {
    stop("x must be a character vector or quanteda tokens object")
  }

  texts_list <- as.list(unname(texts))
  result <- .spacy_env$nlp$get_noun_chunks(texts_list)
  df <- reticulate::py_to_r(result)

  # Map doc_id
  if (nrow(df) > 0 && "doc_id" %in% names(df) && length(doc_names) > 0) {
    doc_id_map <- data.frame(
      old_id = paste0("text", seq_along(doc_names)),
      new_id = doc_names,
      stringsAsFactors = FALSE
    )
    df$doc_id <- doc_id_map$new_id[match(df$doc_id, doc_id_map$old_id)]
  }

  return(df)
}

#' Extract Subjects and Objects
#'
#' @description
#' Extract subject-verb-object (SVO) triples from texts using dependency parsing.
#'
#' @param x Character vector of texts OR a quanteda tokens object.
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data frame with SVO information.
#'
#' @family lexical
#' @export
extract_subjects_objects <- function(x, model = "en_core_web_sm") {
  if (!spacy_initialized() || .spacy_env$model != model) {
    init_spacy_nlp(model)
  }

  if (inherits(x, "tokens")) {
    texts <- vapply(as.list(x), function(toks) paste(toks, collapse = " "), character(1))
    doc_names <- quanteda::docnames(x)
  } else if (is.character(x)) {
    texts <- x
    doc_names <- names(x)
    if (is.null(doc_names)) {
      doc_names <- paste0("text", seq_along(texts))
    }
  } else {
    stop("x must be a character vector or quanteda tokens object")
  }

  texts_list <- as.list(unname(texts))
  result <- .spacy_env$nlp$get_subjects_objects(texts_list)
  df <- reticulate::py_to_r(result)

  if (nrow(df) > 0 && "doc_id" %in% names(df) && length(doc_names) > 0) {
    doc_id_map <- data.frame(
      old_id = paste0("text", seq_along(doc_names)),
      new_id = doc_names,
      stringsAsFactors = FALSE
    )
    df$doc_id <- doc_id_map$new_id[match(df$doc_id, doc_id_map$old_id)]
  }

  return(df)
}

#' Get Sentences
#'
#' @description
#' Segment texts into sentences using spaCy's sentence boundary detection.
#'
#' @param x Character vector of texts OR a quanteda tokens object.
#' @param model Character; spaCy model to use (default: "en_core_web_sm").
#'
#' @return A data frame with sentence information.
#'
#' @family lexical
#' @export
get_sentences <- function(x, model = "en_core_web_sm") {
  if (!spacy_initialized() || .spacy_env$model != model) {
    init_spacy_nlp(model)
  }

  if (inherits(x, "tokens")) {
    texts <- vapply(as.list(x), function(toks) paste(toks, collapse = " "), character(1))
    doc_names <- quanteda::docnames(x)
  } else if (is.character(x)) {
    texts <- x
    doc_names <- names(x)
    if (is.null(doc_names)) {
      doc_names <- paste0("text", seq_along(texts))
    }
  } else {
    stop("x must be a character vector or quanteda tokens object")
  }

  texts_list <- as.list(unname(texts))
  result <- .spacy_env$nlp$get_sentences(texts_list)
  df <- reticulate::py_to_r(result)

  if (nrow(df) > 0 && "doc_id" %in% names(df) && length(doc_names) > 0) {
    doc_id_map <- data.frame(
      old_id = paste0("text", seq_along(doc_names)),
      new_id = doc_names,
      stringsAsFactors = FALSE
    )
    df$doc_id <- doc_id_map$new_id[match(df$doc_id, doc_id_map$old_id)]
  }

  return(df)
}

#' Calculate Word Similarity
#'
#' @description
#' Calculate semantic similarity between two words using word vectors.
#' Requires a spaCy model with word vectors (en_core_web_md or en_core_web_lg).
#'
#' @param word1 Character; first word.
#' @param word2 Character; second word.
#' @param model Character; spaCy model to use (default: "en_core_web_md").
#'
#' @return A list with similarity score and metadata.
#'
#' @family lexical
#' @export
get_word_similarity <- function(word1, word2, model = "en_core_web_md") {
  if (!spacy_initialized() || .spacy_env$model != model) {
    init_spacy_nlp(model)
  }

  result <- .spacy_env$nlp$get_word_similarity(word1, word2)
  as.list(result)
}

#' Find Similar Words
#'
#' @description
#' Find words most similar to a given word using word vectors.
#' Requires a spaCy model with word vectors (en_core_web_md or en_core_web_lg).
#'
#' @param word Character; target word.
#' @param top_n Integer; number of similar words to return (default: 10).
#' @param model Character; spaCy model to use (default: "en_core_web_md").
#'
#' @return A data frame with similar words and similarity scores.
#'
#' @family lexical
#' @export
find_similar_words <- function(word, top_n = 10L, model = "en_core_web_md") {
  if (!spacy_initialized() || .spacy_env$model != model) {
    init_spacy_nlp(model)
  }

  result <- .spacy_env$nlp$find_similar_words(word, as.integer(top_n))
  df <- reticulate::py_to_r(result)

  return(df)
}

#' Get spaCy Word Embeddings
#'
#' @description
#' Get word vector embeddings for words or texts using spaCy.
#' Requires a spaCy model with word vectors.
#'
#' @param texts Character vector of words or texts.
#' @param model Character; spaCy model to use (default: "en_core_web_md").
#'
#' @return A matrix of word embeddings (rows = texts, cols = dimensions).
#'
#' @family lexical
#' @export
get_spacy_embeddings <- function(texts, model = "en_core_web_md") {
  if (!spacy_initialized() || .spacy_env$model != model) {
    init_spacy_nlp(model)
  }

  if (!spacy_has_vectors()) {
    stop("Model '", model, "' has no word vectors. Use en_core_web_md or en_core_web_lg.")
  }

  # Get vectors via Python - access the underlying nlp object
  nlp <- .spacy_env$nlp$nlp

  vectors <- lapply(texts, function(text) {
    doc <- nlp(text)
    as.numeric(doc$vector)
  })

  # Convert to matrix
  mat <- do.call(rbind, vectors)
  rownames(mat) <- texts

  return(mat)
}

#' Parse Morphology String
#'
#' @description
#' Parse spaCy's morphology string format into individual columns.
#' Used internally by morphology analysis functions.
#' Always extracts all common morphology features (Number, Tense, VerbForm,
#' Person, Case, Mood, Aspect) regardless of the features parameter.
#'
#' @param data Data frame with a 'morph' column from spaCy parsing.
#' @param features Character vector of feature names (ignored, kept for
#'   backwards compatibility). All features are always extracted.
#'
#' @return Data frame with additional morph_* columns for each feature.
#'
#' @family lexical
#' @export
parse_morphology_string <- function(data, features = NULL) {
  # Always extract all common morphology features
  all_features <- c("Number", "Tense", "VerbForm", "Person", "Case", "Mood", "Aspect")

  for (feat in all_features) {
    col_name <- paste0("morph_", feat)
    data[[col_name]] <- vapply(data$morph, function(m) {
      if (is.null(m) || length(m) == 0 || !is.atomic(m)) return(NA_character_)
      m_str <- as.character(m)[1]
      if (is.na(m_str) || m_str == "") return(NA_character_)
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
  return(data)
}


################################################################################
# LOG ODDS RATIO ANALYSIS
################################################################################

#' Calculate Log Odds Ratio Between Categories
#'
#' @description
#' Computes log odds ratio to compare word frequencies between categories.
#' Identifies words that are distinctively used in one category vs another.
#' Uses Laplace smoothing to handle zero counts.
#'
#' @param dfm_object A quanteda dfm object
#' @param group_var Character, name of the grouping variable in docvars
#' @param comparison_mode Character, one of "binary", "one_vs_rest", or "pairwise"
#'   \itemize{
#'     \item binary: Compare two categories directly
#'     \item one_vs_rest: Compare each category against all others combined
#'     \item pairwise: Compare all pairs of categories
#'   }
#' @param reference_level Character, reference category for binary comparison (default: first level)
#' @param top_n Number of top terms per comparison (default: 10)
#' @param min_count Minimum word count to include (default: 5)
#'
#' @return Data frame with columns:
#'   \itemize{
#'     \item term: The word/feature
#'     \item category1: First category in comparison
#'     \item category2: Second category in comparison
#'     \item count1: Count in category 1
#'     \item count2: Count in category 2
#'     \item odds1: Odds in category 1
#'     \item odds2: Odds in category 2
#'     \item odds_ratio: Ratio of odds

#'     \item log_odds_ratio: Log of odds ratio (positive = more in compared category)
#'   }
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' library(quanteda)
#' corp <- corpus(c("The cat runs fast", "Dogs are loyal pets",
#'                  "Cats sleep all day", "My dog loves walks"),
#'                docvars = data.frame(animal = c("cat", "dog", "cat", "dog")))
#' dfm <- tokens(corp) %>% dfm()
#' log_odds <- calculate_log_odds_ratio(dfm, "animal")
#' }
calculate_log_odds_ratio <- function(dfm_object,
                                      group_var,
                                      comparison_mode = c("binary", "one_vs_rest", "pairwise"),
                                      reference_level = NULL,
                                      top_n = 10,
                                      min_count = 5) {

  comparison_mode <- match.arg(comparison_mode)

  if (!inherits(dfm_object, "dfm")) {
    stop("dfm_object must be a quanteda dfm object")
  }

  if (!group_var %in% names(quanteda::docvars(dfm_object))) {
    stop("group_var '", group_var, "' not found in document variables")
  }

  groups <- quanteda::docvars(dfm_object)[[group_var]]
  levels <- unique(groups[!is.na(groups)])

  if (length(levels) < 2) {
    stop(
      "Need at least 2 categories for comparison in '", group_var, "'. ",
      "Found only: ", paste(levels, collapse = ", "),
      call. = FALSE
    )
  }

  # Helper function for pairwise comparison
  compare_two <- function(dfm_obj, level1, level2, groups) {
    idx1 <- which(groups == level1)
    idx2 <- which(groups == level2)

    if (length(idx1) == 0 || length(idx2) == 0) {
      return(NULL)
    }

    # Sum counts per group
    counts1 <- Matrix::colSums(dfm_obj[idx1, , drop = FALSE])
    counts2 <- Matrix::colSums(dfm_obj[idx2, , drop = FALSE])

    # Filter by minimum count
    keep <- (counts1 + counts2) >= min_count
    counts1 <- counts1[keep]
    counts2 <- counts2[keep]

    if (length(counts1) == 0) {
      return(NULL)
    }

    # Laplace smoothing (+1)
    total1 <- sum(counts1) + length(counts1)
    total2 <- sum(counts2) + length(counts2)

    odds1 <- (counts1 + 1) / total1
    odds2 <- (counts2 + 1) / total2

    odds_ratio <- odds1 / odds2
    log_odds <- log(odds_ratio)

    result <- data.frame(
      term = names(counts1),
      category1 = level1,
      category2 = level2,
      count1 = as.numeric(counts1),
      count2 = as.numeric(counts2),
      odds1 = odds1,
      odds2 = odds2,
      odds_ratio = odds_ratio,
      log_odds_ratio = log_odds,
      stringsAsFactors = FALSE
    )

    # Get top terms by absolute log odds
    result <- result[order(abs(result$log_odds_ratio), decreasing = TRUE), ]
    utils::head(result, top_n)  # Get top by absolute log odds
  }

  results <- list()


  if (comparison_mode == "binary") {
    if (length(levels) != 2 && is.null(reference_level)) {
      message("More than 2 categories. Using first two: ", levels[1], " vs ", levels[2])
    }

    if (is.null(reference_level)) {
      # Default: second level compared to first (first as reference)
      level1 <- levels[2]  # Compared (numerator)
      level2 <- levels[1]  # Reference (denominator)
    } else {
      # User-specified reference: compare the other level to reference
      level1 <- setdiff(levels, reference_level)[1]  # Compared (numerator)
      level2 <- reference_level  # Reference (denominator)
    }

    results[[1]] <- compare_two(dfm_object, level1, level2, groups)

  } else if (comparison_mode == "one_vs_rest") {
    for (level in levels) {
      other_levels <- setdiff(levels, level)
      # Combine all other categories
      groups_binary <- ifelse(groups == level, level, "Other")
      result <- compare_two(dfm_object, level, "Other", groups_binary)
      if (!is.null(result)) {
        results[[length(results) + 1]] <- result
      }
    }

  } else if (comparison_mode == "pairwise") {
    pairs <- utils::combn(levels, 2, simplify = FALSE)
    for (pair in pairs) {
      result <- compare_two(dfm_object, pair[1], pair[2], groups)
      if (!is.null(result)) {
        results[[length(results) + 1]] <- result
      }
    }
  }

  if (length(results) == 0) {
    return(data.frame(
      term = character(),
      category1 = character(),
      category2 = character(),
      count1 = numeric(),
      count2 = numeric(),
      odds1 = numeric(),
      odds2 = numeric(),
      odds_ratio = numeric(),
      log_odds_ratio = numeric()
    ))
  }

  do.call(rbind, results)
}


#' Calculate Weighted Log Odds Ratio
#'
#' @description
#' Computes weighted log odds ratios using the method from Monroe, Colaresi,
#' and Quinn (2008) "Fightin' Words" via the tidylo package. This method
#' weights log odds by variance (z-score) to identify words that reliably
#' distinguish between groups, accounting for sampling variability.
#'
#' @param dfm_object A quanteda dfm object
#' @param group_var Character, name of the document variable to group by
#' @param top_n Number of top terms to return per group (default: 10)
#' @param min_count Minimum total count for a term to be included (default: 5)
#'
#' @return A data frame with columns: group, feature, n, log_odds_weighted,
#'   and log_odds (from tidylo::bind_log_odds)
#'
#' @references
#' Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words:
#' Lexical feature selection and evaluation for identifying the content of
#' political conflict. Political Analysis, 16(4), 372-403.
#'
#' Silge, J., & Robinson, D. (2017). Text mining with R: A tidy approach.
#' O'Reilly Media. https://www.tidytextmining.com/
#'
#' @family lexical analysis
#' @export
#'
#' @examples
#' \dontrun{
#' weighted_odds <- calculate_weighted_log_odds(dfm, "party", top_n = 15)
#' }
calculate_weighted_log_odds <- function(dfm_object,
                                        group_var,
                                        top_n = 10,
                                        min_count = 5) {

  if (!requireNamespace("tidylo", quietly = TRUE)) {
    stop("Package 'tidylo' is required. Please install it with: install.packages('tidylo')")
  }

  if (!inherits(dfm_object, "dfm")) {
    stop("dfm_object must be a quanteda dfm object")
  }

  if (!group_var %in% names(quanteda::docvars(dfm_object))) {
    stop("group_var '", group_var, "' not found in document variables")
  }

  # Get document variables
  doc_vars <- quanteda::docvars(dfm_object)
  doc_vars$doc_id <- quanteda::docnames(dfm_object)

  # Convert DFM to tidy format
  tidy_data <- quanteda::convert(dfm_object, to = "data.frame")
  names(tidy_data)[1] <- "doc_id"

  tidy_long <- tidyr::pivot_longer(
    tidy_data,
    cols = -"doc_id",
    names_to = "feature",
    values_to = "count"
  )

  # Join with document variables
  tidy_long <- dplyr::left_join(
    tidy_long,
    doc_vars[, c("doc_id", group_var)],
    by = "doc_id"
  )

  # Aggregate by group and feature
  grouped_counts <- tidy_long %>%
    dplyr::group_by(.data[[group_var]], feature) %>%
    dplyr::summarise(n = sum(count), .groups = "drop") %>%
    dplyr::filter(n >= min_count)

  # Apply tidylo weighted log odds
  result <- tidylo::bind_log_odds(
    grouped_counts,
    set = !!rlang::sym(group_var),
    feature = feature,
    n = n
  )

  # Get top terms per group by absolute weighted log odds
  result <- result %>%
    dplyr::group_by(.data[[group_var]]) %>%
    dplyr::slice_max(abs(.data$log_odds_weighted), n = top_n, with_ties = FALSE) %>%
    dplyr::ungroup() %>%
    dplyr::arrange(.data[[group_var]], dplyr::desc(abs(.data$log_odds_weighted)))

  as.data.frame(result)
}


#' Plot Log Odds Ratio
#'
#' @description
#' Creates a horizontal bar plot showing log odds ratios for comparing
#' word usage between categories. Positive values indicate higher usage
#' in the first category, negative in the second.
#'
#' @param log_odds_data Data frame from calculate_log_odds_ratio()
#' @param top_n Number of top terms to show per direction (default: 10)
#' @param facet_by Character, column name to facet by (e.g., "category1" for
#'   one_vs_rest comparisons). NULL for no faceting.
#' @param color_positive Color for positive log odds (default: "#10B981" green)
#' @param color_negative Color for negative log odds (default: "#EF4444" red)
#' @param height Plot height in pixels (default: 600)
#' @param width Plot width in pixels (default: NULL for auto)
#' @param title Plot title (default: "Log Odds Ratio Comparison")
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' log_odds <- calculate_log_odds_ratio(dfm, "category", comparison_mode = "binary")
#' plot_log_odds_ratio(log_odds, top_n = 15)
#' }
plot_log_odds_ratio <- function(log_odds_data,
                                 top_n = 10,
                                 facet_by = NULL,
                                 color_positive = "#10B981",
                                 color_negative = "#EF4444",
                                 height = 600,
                                 width = NULL,
                                 title = "Log Odds Ratio Comparison") {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (is.null(log_odds_data) || nrow(log_odds_data) == 0) {
    return(create_empty_plot_message("No log odds data available"))
  }

  # Get top positive and negative terms
  positive <- log_odds_data[log_odds_data$log_odds_ratio > 0, ]
  negative <- log_odds_data[log_odds_data$log_odds_ratio < 0, ]

  positive <- positive[order(positive$log_odds_ratio, decreasing = TRUE), ]
  negative <- negative[order(negative$log_odds_ratio, decreasing = FALSE), ]

  plot_data <- rbind(
    utils::head(positive, top_n),
    utils::head(negative, top_n)
  )

  if (nrow(plot_data) == 0) {
    return(create_empty_plot_message("No significant differences found"))
  }

  # Order by log odds ratio
  plot_data <- plot_data[order(plot_data$log_odds_ratio), ]
  plot_data$term_ordered <- factor(plot_data$term, levels = plot_data$term)

  # Assign colors
  plot_data$color <- ifelse(plot_data$log_odds_ratio > 0, color_positive, color_negative)

  # Create hover text
  plot_data$hover_text <- paste0(
    "<b>", plot_data$term, "</b><br>",
    "Log Odds: ", round(plot_data$log_odds_ratio, 3), "<br>",
    plot_data$category1, ": ", plot_data$count1, "<br>",
    plot_data$category2, ": ", plot_data$count2
  )

  # Get category labels for subtitle
  cat1 <- unique(plot_data$category1)[1]
  cat2 <- unique(plot_data$category2)[1]
  subtitle <- paste0(
    "<span style='color:", color_negative, ";'>", cat2, " (-)</span> vs ",
    "<span style='color:", color_positive, ";'>", cat1, " (+)</span>"
  )

  p <- plotly::plot_ly(
    data = plot_data,
    x = ~log_odds_ratio,
    y = ~term_ordered,
    type = "bar",
    orientation = "h",
    marker = list(color = ~color),
    hoverinfo = "text",
    hovertext = ~hover_text,
    height = height,
    width = width
  ) %>%
    plotly::layout(
      title = list(
        text = paste0(title, "<br><sub>", subtitle, "</sub>"),
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = "Log Odds Ratio",
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 14, color = "#3B3B3B", family = "Roboto, sans-serif"),
        zeroline = TRUE,
        zerolinecolor = "#CBD5E1",
        zerolinewidth = 2
      ),
      yaxis = list(
        title = "",
        tickfont = list(size = 14, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      margin = list(l = 120, r = 40, t = 80, b = 60),
      hoverlabel = list(
        align = "left",
        font = list(size = 14, color = "white", family = "Roboto, sans-serif"),
        bgcolor = "#0c1f4a"
      ),
      shapes = list(
        list(
          type = "line",
          x0 = 0, x1 = 0,
          y0 = 0, y1 = 1,
          yref = "paper",
          line = list(color = "#94A3B8", width = 1, dash = "dot")
        )
      )
    )

  return(p)
}


#' Plot Weighted Log Odds
#'
#' @description
#' Creates a faceted horizontal bar plot showing weighted log odds for comparing
#' word usage across categories using the Fightin' Words method (Monroe et al. 2008).
#' Each group is displayed in a separate facet showing its most distinctive terms.
#'
#' @param weighted_data Data frame from calculate_weighted_log_odds()
#' @param top_n Number of top terms to show per group (default: 10)
#' @param color_positive Color for positive log odds (default: "#10B981" green)
#' @param color_negative Color for negative log odds (default: "#EF4444" red)
#' @param height Plot height in pixels (default: 600)
#' @param width Plot width in pixels (default: NULL for auto)
#' @param title Plot title (default: "Weighted Log Odds by Group")
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' weighted_odds <- calculate_weighted_log_odds(dfm, "party", top_n = 15)
#' plot_weighted_log_odds(weighted_odds)
#' }
plot_weighted_log_odds <- function(weighted_data,
                                   top_n = 10,
                                   color_positive = "#10B981",
                                   color_negative = "#EF4444",
                                   height = 600,
                                   width = NULL,
                                   title = "Weighted Log Odds by Group") {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (is.null(weighted_data) || nrow(weighted_data) == 0) {
    return(create_empty_plot_message("No weighted log odds data available"))
  }

  # Get group variable name (first column that's not feature/n/log_odds)
  group_col <- setdiff(names(weighted_data), c("feature", "n", "log_odds_weighted", "log_odds"))[1]

  if (is.null(group_col)) {
    return(create_empty_plot_message("Could not identify group column"))
  }

  # Get unique groups
  groups <- unique(weighted_data[[group_col]])

  # Create subplot for each group
  plots <- lapply(groups, function(grp) {
    grp_data <- weighted_data[weighted_data[[group_col]] == grp, ]

    # Get top N by absolute weighted log odds
    grp_data <- grp_data[order(abs(grp_data$log_odds_weighted), decreasing = TRUE), ]
    grp_data <- utils::head(grp_data, top_n)

    # Order by weighted log odds for display
    grp_data <- grp_data[order(grp_data$log_odds_weighted), ]
    grp_data$feature_ordered <- factor(grp_data$feature, levels = grp_data$feature)

    # Assign colors
    grp_data$color <- ifelse(grp_data$log_odds_weighted > 0, color_positive, color_negative)

    # Create hover text
    grp_data$hover_text <- paste0(
      "<b>", grp_data$feature, "</b><br>",
      "Weighted Log Odds: ", round(grp_data$log_odds_weighted, 3), "<br>",
      "Count: ", grp_data$n
    )

    plotly::plot_ly(
      data = grp_data,
      x = ~log_odds_weighted,
      y = ~feature_ordered,
      type = "bar",
      orientation = "h",
      marker = list(color = ~color),
      text = ~hover_text,
      hoverinfo = "text",
      showlegend = FALSE
    ) %>%
      plotly::layout(
        annotations = list(
          list(
            text = as.character(grp),
            x = 0.5,
            y = 1.05,
            xref = "paper",
            yref = "paper",
            showarrow = FALSE,
            font = list(size = 14, color = "#0c1f4a", family = "Roboto, sans-serif")
          )
        ),
        xaxis = list(title = ""),
        yaxis = list(title = ""),
        shapes = list(
          list(
            type = "line",
            x0 = 0, x1 = 0,
            y0 = 0, y1 = 1,
            yref = "paper",
            line = list(color = "#94A3B8", width = 1, dash = "dot")
          )
        )
      )
  })

  # Combine subplots
  n_groups <- length(groups)
  n_rows <- ceiling(n_groups / 2)

  p <- plotly::subplot(
    plots,
    nrows = n_rows,
    shareX = FALSE,
    shareY = FALSE,
    margin = 0.08
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 20, color = "#0c1f4a", family = "Roboto, sans-serif"),
        x = 0.5, xanchor = "center"
      ),
      height = max(height, n_rows * 300),
      margin = list(l = 100, r = 40, t = 80, b = 60),
      hoverlabel = list(
        align = "left",
        font = list(size = 14, color = "white", family = "Roboto, sans-serif"),
        bgcolor = "#0c1f4a"
      )
    )

  return(p)
}


################################################################################
# LEXICAL DISPERSION ANALYSIS
################################################################################

#' Calculate Lexical Dispersion
#'
#' @description
#' Computes lexical dispersion data for specified terms across a corpus.
#' Shows where terms appear within each document, useful for understanding
#' term distribution patterns.
#'
#' @param tokens_object A quanteda tokens object
#' @param terms Character vector of terms to analyze
#' @param scale Character, "relative" (0-1 normalized) or "absolute" (token position)
#'
#' @return Data frame with columns:
#'   \itemize{
#'     \item doc_id: Document identifier
#'     \item term: The search term
#'     \item position: Position in document (relative or absolute)
#'     \item doc_length: Total tokens in document
#'   }
#'
#' @family lexical
#' @export
#'
#' @examples
#' \dontrun{
#' library(quanteda)
#' toks <- tokens(c("The cat sat on the mat", "The dog ran in the park"))
#' dispersion <- calculate_lexical_dispersion(toks, c("the", "cat", "dog"))
#' }
calculate_lexical_dispersion <- function(tokens_object,
                                          terms,
                                          scale = c("relative", "absolute")) {

  scale <- match.arg(scale)


  if (!inherits(tokens_object, "tokens")) {
    stop("tokens_object must be a quanteda tokens object")
  }

  if (is.null(terms) || length(terms) == 0) {
    return(data.frame(
      doc_id = character(),
      term = character(),
      position = numeric(),
      doc_length = integer(),
      stringsAsFactors = FALSE
    ))
  }

  # Convert terms to lowercase for matching

  terms_lower <- tolower(terms)

  results <- list()

  for (i in seq_along(tokens_object)) {
    doc_tokens <- as.character(tokens_object[[i]])
    doc_tokens_lower <- tolower(doc_tokens)
    doc_length <- length(doc_tokens)
    doc_name <- names(tokens_object)[i]

    if (is.null(doc_name) || doc_name == "") {
      doc_name <- paste0("Doc ", i)
    }

    for (term in terms) {
      term_lower <- tolower(term)
      positions <- which(doc_tokens_lower == term_lower)

      if (length(positions) > 0) {
        if (scale == "relative") {
          positions <- positions / doc_length
        }

        results[[length(results) + 1]] <- data.frame(
          doc_id = rep(doc_name, length(positions)),
          term = rep(term, length(positions)),
          position = positions,
          doc_length = rep(doc_length, length(positions)),
          stringsAsFactors = FALSE
        )
      }
    }
  }

  if (length(results) == 0) {
    return(data.frame(
      doc_id = character(),
      term = character(),
      position = numeric(),
      doc_length = integer(),
      stringsAsFactors = FALSE
    ))
  }

  do.call(rbind, results)
}


#' Plot Lexical Dispersion
#'
#' @description
#' Creates an X-ray plot showing where terms appear across documents.
#' Each row represents a term, and marks indicate occurrences.
#'
#' @param dispersion_data Data frame from calculate_lexical_dispersion()
#' @param scale Character, "relative" or "absolute" (must match calculation)
#' @param title Plot title (default: "Lexical Dispersion")
#' @param colors Named vector of colors for each term, or NULL for auto
#' @param height Plot height in pixels (default: 400)
#' @param width Plot width in pixels (default: NULL for auto)
#' @param marker_size Size of position markers (default: 8)
#'
#' @return A plotly object
#'
#' @family visualization
#' @export
#'
#' @examples
#' \dontrun{
#' dispersion <- calculate_lexical_dispersion(tokens, c("education", "technology"))
#' plot_lexical_dispersion(dispersion)
#' }
plot_lexical_dispersion <- function(dispersion_data,
                                     scale = "relative",
                                     title = "Lexical Dispersion",
                                     colors = NULL,
                                     height = 400,
                                     width = NULL,
                                     marker_size = 8) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (is.null(dispersion_data) || nrow(dispersion_data) == 0) {
    return(create_empty_plot_message("No term occurrences found in the corpus"))
  }

  # Get unique terms and assign colors
  unique_terms <- unique(dispersion_data$term)

  if (is.null(colors)) {
    # Default color palette
    default_colors <- c("#3B82F6", "#10B981", "#F59E0B", "#EF4444",
                        "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16")
    colors <- stats::setNames(
      rep(default_colors, length.out = length(unique_terms)),
      unique_terms
    )
  }

  # Create y-axis positions for terms
  dispersion_data$y_pos <- match(dispersion_data$term, unique_terms)

  # Create hover text
  dispersion_data$hover_text <- paste0(
    "<b>", dispersion_data$term, "</b><br>",
    "Document: ", dispersion_data$doc_id, "<br>",
    "Position: ", round(dispersion_data$position, 3),
    if (scale == "relative") " (relative)" else ""
  )

  # Build plot
  p <- plotly::plot_ly(height = height, width = width)

  for (term in unique_terms) {
    term_data <- dispersion_data[dispersion_data$term == term, ]
    term_y <- match(term, unique_terms)

    p <- p %>%
      plotly::add_trace(
        data = term_data,
        x = ~position,
        y = rep(term_y, nrow(term_data)),
        type = "scatter",
        mode = "markers",
        marker = list(
          symbol = "line-ns",
          size = marker_size,
          color = colors[term],
          line = list(width = 2, color = colors[term])
        ),
        name = term,
        hoverinfo = "text",
        hovertext = ~hover_text
      )
  }

  # X-axis label based on scale
  x_label <- if (scale == "relative") {
    "Relative Position (0 = start, 1 = end)"
  } else {
    "Token Position"
  }

  p <- p %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = x_label,
        titlefont = list(size = 14, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 14, color = "#3B3B3B", family = "Roboto, sans-serif"),
        range = if (scale == "relative") c(0, 1) else NULL,
        zeroline = FALSE
      ),
      yaxis = list(
        title = "",
        tickmode = "array",
        tickvals = seq_along(unique_terms),
        ticktext = unique_terms,
        tickfont = list(size = 14, color = "#3B3B3B", family = "Roboto, sans-serif"),
        zeroline = FALSE
      ),
      margin = list(l = 120, r = 40, t = 60, b = 100),
      showlegend = TRUE,
      legend = list(
        orientation = "h",
        x = 0.5,
        xanchor = "center",
        y = -0.25
      ),
      hoverlabel = list(
        align = "left",
        font = list(size = 14, color = "white", family = "Roboto, sans-serif"),
        bgcolor = "#0c1f4a"
      )
    )

  return(p)
}


#' Calculate Dispersion Metrics
#'
#' @description
#' Computes quantitative dispersion metrics for terms, measuring how
#' evenly distributed they are across the corpus.
#'
#' @param tokens_object A quanteda tokens object
#' @param terms Character vector of terms to analyze
#'
#' @return Data frame with columns:
#'   \itemize{
#'     \item term: The search term
#'     \item frequency: Total occurrences
#'     \item doc_count: Number of documents containing term
#'     \item doc_ratio: Proportion of documents containing term
#'     \item juilland_d: Juilland's D dispersion (0-1, higher = more even)
#'     \item rosengren_s: Rosengren's S dispersion
#'   }
#'
#' @family lexical
#' @export
calculate_dispersion_metrics <- function(tokens_object, terms) {

  if (!inherits(tokens_object, "tokens")) {
    stop("tokens_object must be a quanteda tokens object")
  }

  n_docs <- length(tokens_object)
  terms_lower <- tolower(terms)

  results <- lapply(terms, function(term) {
    term_lower <- tolower(term)

    # Count occurrences in each document
    doc_counts <- vapply(tokens_object, function(doc_tokens) {
      sum(tolower(as.character(doc_tokens)) == term_lower)
    }, integer(1))

    total_freq <- sum(doc_counts)
    doc_count <- sum(doc_counts > 0)
    doc_ratio <- doc_count / n_docs

    # Calculate Juilland's D
    if (total_freq > 0 && n_docs > 1) {
      expected <- total_freq / n_docs
      cv <- stats::sd(doc_counts) / mean(doc_counts)  # Coefficient of variation
      juilland_d <- 1 - (cv / sqrt(n_docs - 1))
      juilland_d <- max(0, min(1, juilland_d))  # Clamp to 0-1
    } else {
      juilland_d <- NA_real_
    }

    # Calculate Rosengren's S
    if (total_freq > 0 && n_docs > 1) {
      props <- doc_counts / total_freq
      props <- props[props > 0]  # Only non-zero
      rosengren_s <- exp(sum(props * log(props)) / (-log(n_docs)))
    } else {
      rosengren_s <- NA_real_
    }

    data.frame(
      term = term,
      frequency = total_freq,
      doc_count = doc_count,
      doc_ratio = round(doc_ratio, 3),
      juilland_d = round(juilland_d, 3),
      rosengren_s = round(rosengren_s, 3),
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, results)
}
