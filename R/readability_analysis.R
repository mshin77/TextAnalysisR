# Package-level cache for expensive lexical diversity computations
.lexdiv_cache <- new.env(hash = TRUE, parent = emptyenv())

#' Clear Lexical Diversity Cache
#'
#' @description
#' Clears the internal cache used for lexical diversity calculations.
#' Call this function if you need to free memory or ensure fresh calculations.
#'
#' @return Invisible NULL
#' @export
clear_lexdiv_cache <- function() {
  rm(list = ls(.lexdiv_cache), envir = .lexdiv_cache)
  invisible(NULL)
}

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


#' Lexical Diversity Analysis
#'
#' @description
#' Calculates lexical diversity metrics to measure vocabulary richness.
#' MTLD and MATTR are most stable and text-length independent.
#'
#' @param x A tokens object (preferred) or document-feature matrix from quanteda.
#'   For accurate MTLD calculation, pass a tokens object or provide the `texts` parameter.
#'   DFM input loses token order, which affects MTLD accuracy (McCarthy & Jarvis, 2010).
#' @param measures Character vector of measures to calculate. Options:
#'   "all", "MTLD" (recommended), "MATTR" (recommended), "MSTTR", "TTR", "CTTR", "Maas", "K", "D"
#' @param texts Optional character vector of original texts. Required for accurate MTLD
#'   when passing a DFM (since DFM loses token order). Also used for average sentence length.
#' @param cache_key Optional character string for caching expensive computations.
#'   When provided, results are cached using this key and retrieved on subsequent calls
#'   with the same key. Use `clear_lexdiv_cache()` to clear the cache.
#'
#' @return A list with lexical_diversity (data frame) and summary_stats
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
        # Uses O(n) environment-based hash tracking instead of O(n²) vector concatenation
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
