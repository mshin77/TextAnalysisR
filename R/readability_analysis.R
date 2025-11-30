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
        font = list(size = 20, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        tickfont = list(size = 18, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        tickfont = list(size = 18, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
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
        font = list(size = 18, color = "#0c1f4a", family = "Montserrat, sans-serif")
      ),
      xaxis = list(
        title = list(text = "Document"),
        tickangle = text_angle,
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Montserrat, sans-serif")
      ),
      yaxis = list(
        title = list(text = metric),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Montserrat, sans-serif")
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
#' @param dfm_object A document-feature matrix or tokens object from quanteda
#' @param measures Character vector of measures to calculate. Options:
#'   "all", "MTLD" (recommended), "MATTR" (recommended), "MSTTR", "TTR", "CTTR", "Maas", "K", "D"
#' @param texts Optional character vector of original texts for calculating average sentence length
#'
#' @return A list with lexical_diversity (data frame) and summary_stats
#'
#' @export
#'
#' @examples
#' \dontrun{
#' data(SpecialEduTech)
#' texts <- SpecialEduTech$abstract[1:10]
#' corp <- quanteda::corpus(texts)
#' toks <- quanteda::tokens(corp)
#' dfm_obj <- quanteda::dfm(toks)
#' lex_div <- lexical_diversity_analysis(dfm_obj, texts = texts)
#' print(lex_div)
#' }
#'
#' @importFrom quanteda.textstats textstat_lexdiv
#' @importFrom quanteda docnames
lexical_diversity_analysis <- function(dfm_object,
                                      measures = "all",
                                      texts = NULL) {

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

  tryCatch({
    # Calculate minimum document length to set appropriate window size
    doc_lengths <- quanteda::ntoken(dfm_object)
    min_length <- min(doc_lengths)

    # Set window size for MATTR/MSTTR (default is 100, adjust if documents are shorter)
    window_size <- min(100, max(10, min_length))

    # Calculate quanteda measures if any requested
    if (length(measures_to_use) > 0) {
      lexdiv_results <- suppressWarnings(
        quanteda.textstats::textstat_lexdiv(
          dfm_object,
          measure = measures_to_use,
          MATTR_window = window_size,
          MSTTR_segment = window_size
        )
      )
    } else {
      # Create empty data frame with document names
      lexdiv_results <- data.frame(document = quanteda::docnames(dfm_object))
    }

    # Add MTLD if requested - use custom implementation (McCarthy & Jarvis 2010)
    if (mtld_requested) {
      tryCatch({
        # Custom MTLD implementation based on McCarthy & Jarvis (2010)
        calculate_mtld <- function(tokens, factor_size = 0.72) {
          if (length(tokens) < 10) return(NA_real_)

          # Forward MTLD
          forward_factors <- 0
          current_ttr <- 1
          factor_tokens <- c()

          for (token in tokens) {
            factor_tokens <- c(factor_tokens, token)
            current_ttr <- length(unique(factor_tokens)) / length(factor_tokens)

            if (current_ttr <= factor_size) {
              forward_factors <- forward_factors + 1
              factor_tokens <- c()
              current_ttr <- 1
            }
          }

          # Add partial factor
          if (length(factor_tokens) > 0) {
            partial <- (1 - current_ttr) / (1 - factor_size)
            forward_factors <- forward_factors + partial
          }

          forward_mtld <- if (forward_factors > 0) length(tokens) / forward_factors else NA_real_

          # Backward MTLD
          tokens_rev <- rev(tokens)
          backward_factors <- 0
          factor_tokens <- c()

          for (token in tokens_rev) {
            factor_tokens <- c(factor_tokens, token)
            current_ttr <- length(unique(factor_tokens)) / length(factor_tokens)

            if (current_ttr <= factor_size) {
              backward_factors <- backward_factors + 1
              factor_tokens <- c()
              current_ttr <- 1
            }
          }

          if (length(factor_tokens) > 0) {
            partial <- (1 - current_ttr) / (1 - factor_size)
            backward_factors <- backward_factors + partial
          }

          backward_mtld <- if (backward_factors > 0) length(tokens) / backward_factors else NA_real_

          # Average forward and backward
          mtld <- mean(c(forward_mtld, backward_mtld), na.rm = TRUE)
          return(mtld)
        }

        # Get tokens for each document - handle both tokens and dfm objects
        mtld_values <- sapply(seq_len(quanteda::ndoc(dfm_object)), function(i) {
          # Check if input is a tokens object or dfm
          if (inherits(dfm_object, "tokens")) {
            # Direct access to token sequence (preserves order)
            doc_tokens <- as.character(dfm_object[[i]])
          } else {
            # Extract from DFM (reconstruct sequence - order not preserved)
            doc_row <- dfm_object[i, ]
            token_counts <- as.vector(doc_row)
            token_names <- quanteda::featnames(dfm_object)
            non_zero <- token_counts > 0
            if (sum(non_zero) < 10) return(NA_real_)
            doc_tokens <- rep(token_names[non_zero], token_counts[non_zero])
          }

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
      lexdiv_results$document <- quanteda::docnames(dfm_object)
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

    return(list(
      lexical_diversity = lexdiv_results,
      summary_stats = summary_stats
    ))

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

  all_scores <- list()
  for (i in seq_along(valid_metrics)) {
    metric <- valid_metrics[i]
    measure_name <- mapped_metrics[i]

    tryCatch({
      score <- quanteda.textstats::textstat_readability(corp, measure = measure_name)
      all_scores[[metric]] <- score[[2]]
    }, error = function(e) {
      warning(paste("Could not calculate", metric, ":", e$message))
      all_scores[[metric]] <<- rep(NA, length(texts))
    })
  }

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
#' @export
lexical_frequency_analysis <- function(...) {
  return(plot_word_frequency(...))
}
