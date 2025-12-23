#' Analyze Text Sentiment
#'
#' @description
#' Performs sentiment analysis on text data using the syuzhet package.
#' Returns sentiment scores and classifications.
#'
#' @param texts Character vector of texts to analyze
#' @param method Sentiment analysis method: "syuzhet", "bing", "afinn", or "nrc" (default: "syuzhet")
#' @param doc_ids Optional character vector of document identifiers (default: NULL)
#'
#' @return A data frame with columns:
#'   \describe{
#'     \item{document}{Document identifier}
#'     \item{text}{Original text}
#'     \item{sentiment_score}{Numeric sentiment score}
#'     \item{sentiment}{Classification: "positive", "negative", or "neutral"}
#'   }
#'
#' @family sentiment
#' @export
#'
#' @examples
#' \dontrun{
#' texts <- c(
#'   "This research shows promising results for students.",
#'   "The intervention had no significant effect.",
#'   "Students struggled with the complex material."
#' )
#' results <- analyze_sentiment(texts)
#' print(results)
#' }
analyze_sentiment <- function(texts,
                              method = "syuzhet",
                              doc_ids = NULL) {

  if (!requireNamespace("syuzhet", quietly = TRUE)) {
    stop("Package 'syuzhet' is required. Please install it with: install.packages('syuzhet')")
  }

  if (is.null(doc_ids)) {
    doc_ids <- paste0("doc", seq_along(texts))
  }

  sentiment_scores <- syuzhet::get_sentiment(texts, method = method)

  sentiment_classification <- ifelse(
    sentiment_scores > 0, "positive",
    ifelse(sentiment_scores < 0, "negative", "neutral")
  )

  data.frame(
    document = doc_ids,
    text = texts,
    sentiment_score = sentiment_scores,
    sentiment = sentiment_classification,
    stringsAsFactors = FALSE
  )
}


#' Plot Sentiment Distribution
#'
#' @description
#' Creates a bar plot showing the distribution of sentiment classifications.
#'
#' @param sentiment_data Data frame from analyze_sentiment() or with 'sentiment' column
#' @param title Plot title (default: "Sentiment Distribution")
#'
#' @return A plotly bar chart
#'
#' @family sentiment
#' @export
#'
#' @examples
#' \dontrun{
#' texts <- c("Great results!", "Poor performance", "Okay outcome")
#' sentiment_data <- analyze_sentiment(texts)
#' plot <- plot_sentiment_distribution(sentiment_data)
#' print(plot)
#' }
plot_sentiment_distribution <- function(sentiment_data,
                                        title = "Sentiment Distribution") {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (!"sentiment" %in% names(sentiment_data)) {
    stop("Data must contain a 'sentiment' column. Use analyze_sentiment() first.")
  }

  sentiment_counts <- table(sentiment_data$sentiment)

  ordered_sentiments <- c("positive", "negative", "neutral")
  sentiment_counts <- sentiment_counts[ordered_sentiments[ordered_sentiments %in% names(sentiment_counts)]]

  colors <- get_sentiment_colors()

  plotly::plot_ly(
    x = names(sentiment_counts),
    y = as.numeric(sentiment_counts),
    type = "bar",
    text = as.numeric(sentiment_counts),
    textposition = "none",
    marker = list(color = colors[names(sentiment_counts)]),
    hovertemplate = "%{x}<br>Count: %{y}<extra></extra>"
  ) %>%
    apply_standard_plotly_layout(
      title = title,
      xaxis_title = "Sentiment",
      yaxis_title = "Number of Documents"
    )
}


#' Plot Sentiment by Category
#'
#' @description
#' Creates a grouped or stacked bar plot showing sentiment distribution across categories.
#'
#' @param sentiment_data Data frame with 'sentiment' column
#' @param category_var Name of the category variable column
#' @param plot_type Type of plot: "bar" or "stacked" (default: "bar")
#' @param title Plot title (default: auto-generated)
#'
#' @return A plotly grouped/stacked bar chart
#'
#' @family sentiment
#' @export
#'
#' @examples
#' \dontrun{
#' data <- data.frame(
#'   text = c("Good", "Bad", "Okay", "Great", "Poor"),
#'   category = c("A", "A", "B", "B", "B")
#' )
#' data <- cbind(data, analyze_sentiment(data$text))
#' plot <- plot_sentiment_by_category(data, "category")
#' print(plot)
#' }
plot_sentiment_by_category <- function(sentiment_data,
                                       category_var,
                                       plot_type = "bar",
                                       title = NULL) {

  if (!requireNamespace("plotly", quietly = TRUE) || !requireNamespace("dplyr", quietly = TRUE)) {
    stop("Packages 'plotly' and 'dplyr' are required.")
  }

  if (!category_var %in% names(sentiment_data)) {
    stop(paste("Category variable", category_var, "not found in data"))
  }

  if (!"sentiment" %in% names(sentiment_data)) {
    stop("Data must contain a 'sentiment' column. Use analyze_sentiment() first.")
  }

  grouped_data <- sentiment_data %>%
    dplyr::group_by(!!rlang::sym(category_var), sentiment) %>%
    dplyr::summarise(count = dplyr::n(), .groups = "drop") %>%
    dplyr::group_by(!!rlang::sym(category_var)) %>%
    dplyr::mutate(proportion = count / sum(count)) %>%
    dplyr::ungroup()

  names(grouped_data)[1] <- "category_var"

  colors <- get_sentiment_colors()

  if (is.null(title)) {
    title <- paste("Sentiment by", category_var)
  }

  plotly::plot_ly(
    grouped_data,
    x = ~category_var,
    y = ~proportion,
    color = ~sentiment,
    colors = colors,
    type = "bar",
    text = ~paste(
      "Category:", category_var,
      "<br>Sentiment:", sentiment,
      "<br>Proportion:", sprintf("%.3f", proportion)
    ),
    hovertemplate = "%{text}<extra></extra>"
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = list(text = category_var),
        tickangle = -45,
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = "Proportion"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      barmode = if (plot_type == "stacked") "stack" else "group",
      font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
      hoverlabel = list(align = "left", font = list(size = 16)),
      margin = list(l = 80, r = 40, t = 80, b = 120)
    ) %>%
    plotly::config(displayModeBar = TRUE)
}


#' Plot Document Sentiment Trajectory
#'
#' @description
#' Creates a line chart showing sentiment scores across documents with color gradient.
#'
#' @param sentiment_data Data frame from analyze_sentiment() with sentiment_score column
#' @param top_n Number of documents to display (default: NULL for all)
#' @param doc_ids Optional vector of custom document IDs for display (default: NULL)
#' @param text_preview Optional vector of text snippets for tooltips (default: NULL)
#' @param title Plot title (default: "Document Sentiment Scores")
#'
#' @return A plotly line chart with color gradient
#'
#' @family sentiment
#' @export
plot_document_sentiment_trajectory <- function(sentiment_data,
                                               top_n = NULL,
                                               doc_ids = NULL,
                                               text_preview = NULL,
                                               title = "Document Sentiment Scores") {

  doc_data <- sentiment_data %>%
    dplyr::arrange(document) %>%
    dplyr::mutate(doc_index = dplyr::row_number())

  if (!is.null(top_n)) {
    doc_data <- doc_data %>%
      dplyr::slice_head(n = top_n)
  }

  if (!is.null(doc_ids)) {
    doc_data$display_id <- doc_ids[doc_data$document]

    if (!is.null(text_preview)) {
      text_content <- text_preview[doc_data$document]
      hover_text <- paste0(
        "<b>Document ID:</b> ", doc_data$display_id, "\n",
        "<b>Sentiment:</b> ", doc_data$sentiment, "\n",
        "<b>Score:</b> ", round(doc_data$sentiment_score, 2),
        ifelse(!is.na(text_content) & text_content != "", paste0("\n<b>Text:</b>\n", text_content), "")
      )
    } else {
      hover_text <- paste0(
        "<b>Document ID:</b> ", doc_data$display_id, "\n",
        "<b>Sentiment:</b> ", doc_data$sentiment, "\n",
        "<b>Score:</b> ", round(doc_data$sentiment_score, 2)
      )
    }
  } else {
    doc_data$display_id <- paste("Doc", doc_data$document)

    if (!is.null(text_preview)) {
      text_content <- text_preview[doc_data$document]
      hover_text <- paste0(
        "<b>Document:</b> ", doc_data$display_id, "<br>",
        "<b>Sentiment:</b> ", doc_data$sentiment, "<br>",
        "<b>Score:</b> ", round(doc_data$sentiment_score, 2),
        ifelse(!is.na(text_content) & text_content != "", paste0("<br><b>Text:</b> ", text_content), "")
      )
    } else {
      hover_text <- paste0(
        "<b>Document:</b> ", doc_data$display_id, "<br>",
        "<b>Sentiment:</b> ", doc_data$sentiment, "<br>",
        "<b>Score:</b> ", round(doc_data$sentiment_score, 2)
      )
    }
  }

  hover_bg_colors <- sapply(doc_data$sentiment_score, get_sentiment_color)

  plotly::plot_ly(
    doc_data,
    x = ~doc_index,
    y = ~sentiment_score,
    type = "scatter",
    mode = "lines+markers",
    text = hover_text,
    hovertemplate = "%{text}<extra></extra>",
    marker = list(
      color = ~sentiment_score,
      colorscale = list(
        c(0, "rgb(239, 68, 68)"),
        c(0.5, "rgb(107, 114, 128)"),
        c(1, "rgb(16, 185, 129)")
      ),
      showscale = TRUE,
      colorbar = list(title = "Sentiment Score")
    ),
    hoverlabel = list(
      bgcolor = hover_bg_colors,
      bordercolor = hover_bg_colors,
      font = list(
        family = "Roboto, sans-serif",
        size = 15,
        color = "#ffffff"
      ),
      align = "left",
      namelength = -1,
      maxwidth = 400
    )
  ) %>%
    apply_standard_plotly_layout(
      title = title,
      xaxis_title = "Document Index",
      yaxis_title = "Sentiment Score"
    ) %>%
    plotly::layout(
      yaxis = list(zeroline = TRUE)
    )
}

#' Analyze Sentiment Using Tidytext Lexicons
#'
#' @description
#' Performs lexicon-based sentiment analysis on a DFM object using tidytext lexicons.
#' Supports AFINN, Bing, and NRC lexicons with comprehensive scoring and emotion analysis.
#' Now supports n-grams for improved negation and intensifier handling.
#'
#' @param dfm_object A quanteda DFM object (unigram or n-gram)
#' @param lexicon Lexicon to use: "afinn", "bing", or "nrc" (default: "afinn")
#' @param texts_df Optional data frame with original texts and metadata (default: NULL)
#' @param feature_type Feature space: "words" (unigrams) or "ngrams" (default: "words")
#' @param ngram_range N-gram size when feature_type = "ngrams" (default: 2 for bigrams)
#' @param texts Optional character vector of texts for n-gram creation (default: NULL)
#'
#' @return A list containing:
#'   \describe{
#'     \item{document_sentiment}{Data frame with sentiment scores per document}
#'     \item{emotion_scores}{Data frame with emotion scores (NRC only)}
#'     \item{summary_stats}{List of summary statistics}
#'     \item{feature_type}{Feature type used for analysis}
#'   }
#'
#' @importFrom tidytext tidy get_sentiments
#' @importFrom dplyr inner_join group_by summarise mutate case_when ungroup n_distinct
#' @importFrom tidyr pivot_wider pivot_longer
#' @family sentiment
#' @export
#'
#' @examples
#' \dontrun{
#' corp <- quanteda::corpus(c("I love this!", "I hate that", "It's okay"))
#' dfm_obj <- quanteda::dfm(quanteda::tokens(corp))
#' results <- sentiment_lexicon_analysis(dfm_obj, lexicon = "afinn")
#' print(results$document_sentiment)
#'
#' texts <- c("not good at all", "very happy indeed")
#' results_ngram <- sentiment_lexicon_analysis(
#'   dfm_obj,
#'   lexicon = "bing",
#'   feature_type = "ngrams",
#'   ngram_range = 2,
#'   texts = texts
#' )
#' }
sentiment_lexicon_analysis <- function(dfm_object,
                                       lexicon = "afinn",
                                       texts_df = NULL,
                                       feature_type = "words",
                                       ngram_range = 2,
                                       texts = NULL) {

  if (!requireNamespace("tidytext", quietly = TRUE)) {
    stop("Package 'tidytext' is required. Please install it.")
  }

  if (feature_type == "ngrams" && !is.null(texts)) {
    if (!requireNamespace("quanteda", quietly = TRUE)) {
      stop("Package 'quanteda' is required for n-gram analysis.")
    }

    message("Creating ", ngram_range, "-gram DFM for sentiment analysis (handles negation and intensifiers)")

    corp <- quanteda::corpus(texts)
    toks <- quanteda::tokens(corp, remove_punct = TRUE, remove_symbols = TRUE)
    toks_ngrams <- quanteda::tokens_ngrams(toks, n = ngram_range)
    dfm_object <- quanteda::dfm(toks_ngrams)
  }

  tidy_dfm <- tidytext::tidy(dfm_object)
  lexicon_name <- tolower(lexicon)

  sentiment_lexicon <- switch(
    lexicon_name,
    "afinn" = tidytext::get_sentiments("afinn"),
    "bing" = tidytext::get_sentiments("bing"),
    "nrc" = tidytext::get_sentiments("nrc"),
    stop("Invalid lexicon. Choose 'afinn', 'bing', or 'nrc'.")
  )

  doc_names <- quanteda::docnames(dfm_object)

  if (lexicon_name == "afinn") {
    doc_sentiment <- tidy_dfm %>%
      dplyr::inner_join(sentiment_lexicon, by = c("term" = "word"), relationship = "many-to-many") %>%
      dplyr::group_by(document) %>%
      dplyr::summarise(
        sentiment_score = sum(value * count, na.rm = TRUE),
        n_words = sum(count),
        avg_sentiment = sentiment_score / n_words,
        sentiment = dplyr::case_when(
          avg_sentiment > 0.5 ~ "positive",
          avg_sentiment < -0.5 ~ "negative",
          TRUE ~ "neutral"
        ),
        .groups = "drop"
      )
  } else if (lexicon_name == "bing") {
    doc_sentiment <- tidy_dfm %>%
      dplyr::inner_join(sentiment_lexicon, by = c("term" = "word"), relationship = "many-to-many") %>%
      dplyr::group_by(document, sentiment) %>%
      dplyr::summarise(n = sum(count), .groups = "drop") %>%
      tidyr::pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)

    if (!("positive" %in% names(doc_sentiment))) {
      doc_sentiment$positive <- 0
    }
    if (!("negative" %in% names(doc_sentiment))) {
      doc_sentiment$negative <- 0
    }

    doc_sentiment <- doc_sentiment %>%
      dplyr::mutate(
        sentiment_score = positive - negative,
        total_sentiment_words = positive + negative,
        sentiment = dplyr::case_when(
          sentiment_score > 0 ~ "positive",
          sentiment_score < 0 ~ "negative",
          TRUE ~ "neutral"
        )
      )
  } else if (lexicon_name == "nrc") {
    doc_sentiment <- tidy_dfm %>%
      dplyr::inner_join(sentiment_lexicon, by = c("term" = "word"), relationship = "many-to-many") %>%
      dplyr::group_by(document, sentiment) %>%
      dplyr::summarise(n = sum(count), .groups = "drop") %>%
      tidyr::pivot_wider(names_from = sentiment, values_from = n, values_fill = 0)

    if ("positive" %in% names(doc_sentiment) && "negative" %in% names(doc_sentiment)) {
      doc_sentiment <- doc_sentiment %>%
        dplyr::mutate(
          sentiment_score = positive - negative,
          sentiment = dplyr::case_when(
            sentiment_score > 0 ~ "positive",
            sentiment_score < 0 ~ "negative",
            TRUE ~ "neutral"
          )
        )
    }
  }

  emotion_data <- NULL
  if (lexicon_name == "nrc") {
    emotion_cols <- c("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")
    available_emotions <- intersect(emotion_cols, names(doc_sentiment))
    if (length(available_emotions) > 0) {
      emotion_data <- doc_sentiment %>%
        dplyr::select(document, dplyr::all_of(available_emotions)) %>%
        tidyr::pivot_longer(cols = -document, names_to = "emotion", values_to = "score") %>%
        dplyr::group_by(emotion) %>%
        dplyr::summarise(total_score = sum(score, na.rm = TRUE), .groups = "drop")
    }
  }

  total_docs_in_corpus <- length(doc_names)
  docs_analyzed <- dplyr::n_distinct(doc_sentiment$document)

  summary_stats <- list(
    total_documents = total_docs_in_corpus,
    documents_analyzed = docs_analyzed,
    documents_without_sentiment = total_docs_in_corpus - docs_analyzed,
    coverage_percentage = round((docs_analyzed / total_docs_in_corpus) * 100, 1),
    positive_docs = sum(doc_sentiment$sentiment == "positive", na.rm = TRUE),
    negative_docs = sum(doc_sentiment$sentiment == "negative", na.rm = TRUE),
    neutral_docs = sum(doc_sentiment$sentiment == "neutral", na.rm = TRUE),
    avg_sentiment_score = mean(doc_sentiment$sentiment_score, na.rm = TRUE)
  )

  list(
    document_sentiment = doc_sentiment,
    emotion_scores = emotion_data,
    summary_stats = summary_stats,
    lexicon_used = lexicon_name,
    feature_type = feature_type
  )
}


#' Embedding-based Sentiment Analysis
#'
#' @description
#' Performs sentiment analysis using transformer-based embeddings and neural models.
#' This approach uses pre-trained language models for contextual sentiment detection
#' without requiring sentiment lexicons. Particularly effective for handling:
#' - Complex contextual sentiment
#' - Implicit sentiment and sarcasm
#' - Domain-specific sentiment
#' - Negation and intensifiers (automatically handled by the model)
#'
#' @param texts Character vector of texts to analyze
#' @param embeddings Optional pre-computed embedding matrix (from generate_embeddings)
#' @param model_name Sentiment model name (default: "distilbert-base-uncased-finetuned-sst-2-english")
#' @param doc_names Optional document names/IDs
#' @param use_gpu Whether to use GPU if available (default: FALSE)
#'
#' @return A list containing:
#'   \describe{
#'     \item{document_sentiment}{Data frame with document-level sentiment scores and classifications}
#'     \item{emotion_scores}{NULL (emotion detection not currently supported for embeddings)}
#'     \item{summary_stats}{Summary statistics including document counts and average scores}
#'     \item{model_used}{Name of the transformer model used}
#'     \item{feature_type}{"embeddings"}
#'   }
#'
#' @family sentiment
#' @export
#'
#' @examples
#' \dontrun{
#' texts <- c(
#'   "The results significantly improved student outcomes.",
#'   "The intervention showed no clear benefit.",
#'   "Students reported difficulty with the material."
#' )
#' result <- sentiment_embedding_analysis(texts)
#' print(result$document_sentiment)
#' print(result$summary_stats)
#' }
sentiment_embedding_analysis <- function(texts,
                                        embeddings = NULL,
                                        model_name = "distilbert-base-uncased-finetuned-sst-2-english",
                                        doc_names = NULL,
                                        use_gpu = FALSE) {

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("Package 'reticulate' is required for embedding-based sentiment analysis. Please install it.")
  }

  if (is.null(doc_names)) {
    doc_names <- paste0("text", seq_along(texts))
  }

  if (length(texts) != length(doc_names)) {
    stop("Length of texts and doc_names must match")
  }

  tryCatch({
    transformers <- reticulate::import("transformers")

    message("Loading sentiment model: ", model_name)

    device <- if (use_gpu) 0L else -1L

    sentiment_pipeline <- transformers$pipeline(
      "sentiment-analysis",
      model = model_name,
      device = device
    )

    message("Analyzing sentiment for ", length(texts), " documents...")

    results <- sentiment_pipeline(texts, truncation = TRUE, max_length = 512L)

    doc_sentiment <- data.frame(
      document = doc_names,
      label = sapply(results, function(x) tolower(x$label)),
      confidence = sapply(results, function(x) x$score),
      stringsAsFactors = FALSE
    )

    doc_sentiment$sentiment_score <- ifelse(
      doc_sentiment$label == "positive",
      doc_sentiment$confidence,
      -doc_sentiment$confidence
    )

    doc_sentiment$sentiment <- doc_sentiment$label

    if ("neg" %in% doc_sentiment$label || "negative" %in% doc_sentiment$label) {
      doc_sentiment$sentiment <- ifelse(
        doc_sentiment$label %in% c("neg", "negative"),
        "negative",
        "positive"
      )
    }

    summary_stats <- list(
      total_documents = length(texts),
      documents_analyzed = nrow(doc_sentiment),
      documents_without_sentiment = 0,
      coverage_percentage = 100,
      positive_docs = sum(doc_sentiment$sentiment == "positive", na.rm = TRUE),
      negative_docs = sum(doc_sentiment$sentiment == "negative", na.rm = TRUE),
      neutral_docs = sum(doc_sentiment$sentiment == "neutral", na.rm = TRUE),
      avg_sentiment_score = mean(doc_sentiment$sentiment_score, na.rm = TRUE),
      avg_confidence = mean(doc_sentiment$confidence, na.rm = TRUE)
    )

    message("Sentiment analysis completed successfully")

    return(list(
      document_sentiment = doc_sentiment,
      emotion_scores = NULL,
      summary_stats = summary_stats,
      model_used = model_name,
      feature_type = "embeddings"
    ))

  }, error = function(e) {
    stop(
      "Error in embedding-based sentiment analysis: ", e$message, "\n",
      "Please ensure Python transformers library is installed:\n",
      "  reticulate::py_install('transformers')\n",
      "  reticulate::py_install('torch')"
    )
  })
}


#' Plot Emotion Radar Chart
#'
#' @description
#' Creates a polar/radar chart for NRC emotion analysis with optional grouping.
#'
#' @param emotion_data Data frame with emotion scores (columns: emotion, total_score)
#' @param group_var Optional grouping variable column name for overlaid radars (default: NULL)
#' @param normalize Logical, whether to normalize scores to 0-100 scale (default: FALSE)
#' @param title Plot title (default: "Emotion Analysis")
#'
#' @return A plotly polar chart
#'
#' @family sentiment
#' @export
plot_emotion_radar <- function(emotion_data,
                               group_var = NULL,
                               normalize = FALSE,
                               title = "Emotion Analysis") {

  if (!is.null(group_var) && group_var %in% names(emotion_data)) {

    plot_data <- emotion_data

    if (normalize) {
      plot_data <- plot_data %>%
        dplyr::group_by(!!rlang::sym(group_var)) %>%
        dplyr::mutate(
          total_score = if (max(total_score, na.rm = TRUE) > 0) {
            (total_score / max(total_score, na.rm = TRUE)) * 100
          } else {
            total_score
          }
        ) %>%
        dplyr::ungroup()
    }

    categories <- unique(plot_data[[group_var]])

    p <- plotly::plot_ly()

    for (cat in categories) {
      cat_data <- plot_data %>% dplyr::filter(!!rlang::sym(group_var) == cat)

      p <- p %>%
        plotly::add_trace(
          type = 'scatterpolar',
          mode = 'lines+markers',
          r = cat_data$total_score,
          theta = cat_data$emotion,
          fill = 'toself',
          name = as.character(cat)
        )
    }

    p %>%
      plotly::layout(
        title = list(
          text = title,
          font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
        ),
        polar = list(
          radialaxis = list(
            visible = TRUE,
            range = c(0, max(plot_data$total_score, na.rm = TRUE) * 1.1),
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
          ),
          angularaxis = list(
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
          )
        ),
        font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
        hoverlabel = list(
          align = "left",
          font = list(size = 16, family = "Roboto, sans-serif"),
          maxwidth = 300
        ),
        legend = list(
          font = list(size = 16, family = "Roboto, sans-serif")
        ),
        showlegend = TRUE,
        margin = list(l = 80, r = 80, t = 80, b = 80)
      ) %>%
      plotly::config(displayModeBar = TRUE)

  } else {

    scores <- emotion_data$total_score

    if (normalize) {
      max_score <- max(scores, na.rm = TRUE)
      if (max_score > 0) {
        scores <- (scores / max_score) * 100
      }
    }

    plotly::plot_ly(
      type = 'scatterpolar',
      mode = 'lines+markers',
      r = scores,
      theta = emotion_data$emotion,
      fill = 'toself',
      name = 'Emotion Scores',
      marker = list(color = "#8B5CF6")
    ) %>%
      plotly::layout(
        title = list(
          text = title,
          font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
        ),
        polar = list(
          radialaxis = list(
            visible = TRUE,
            range = c(0, max(scores, na.rm = TRUE) * 1.1),
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
          ),
          angularaxis = list(
            tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
          )
        ),
        font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
        hoverlabel = list(
          align = "left",
          font = list(size = 16, family = "Roboto, sans-serif"),
          maxwidth = 300
        ),
        showlegend = FALSE,
        margin = list(l = 80, r = 80, t = 80, b = 80)
      ) %>%
      plotly::config(displayModeBar = TRUE)
  }
}


#' Plot Sentiment Box Plot by Category
#'
#' @description
#' Creates a box plot showing sentiment score distribution by category.
#'
#' @param sentiment_data Data frame from analyze_sentiment() containing sentiment_score
#'   and category columns
#' @param category_var Name of the category variable column (default: "category_var")
#' @param title Plot title (default: "Sentiment Score Distribution")
#'
#' @return A plotly box plot
#'
#' @family sentiment
#' @export
plot_sentiment_boxplot <- function(sentiment_data,
                                   category_var = "category_var",
                                   title = "Sentiment Score Distribution") {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (!category_var %in% names(sentiment_data)) {
    stop("Category variable '", category_var, "' not found in data")
  }

  if (!"sentiment_score" %in% names(sentiment_data)) {
    stop("sentiment_score column not found in data")
  }

  plotly::plot_ly(
    sentiment_data,
    x = as.formula(paste0("~", category_var)),
    y = ~sentiment_score,
    type = "box",
    color = as.formula(paste0("~", category_var))
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = list(text = category_var),
        tickangle = -45,
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = "Sentiment Score"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
      hoverlabel = list(
        align = "left",
        font = list(size = 16, family = "Roboto, sans-serif"),
        maxwidth = 300
      ),
      legend = list(
        font = list(size = 16, family = "Roboto, sans-serif")
      ),
      showlegend = FALSE,
      margin = list(l = 80, r = 40, t = 80, b = 120)
    ) %>%
    plotly::config(displayModeBar = TRUE)
}


#' Plot Sentiment Violin Plot by Category
#'
#' @description
#' Creates a violin plot showing sentiment score distribution by category.
#'
#' @param sentiment_data Data frame from analyze_sentiment() containing sentiment_score
#'   and category columns
#' @param category_var Name of the category variable column (default: "category_var")
#' @param title Plot title (default: "Sentiment Score Distribution")
#'
#' @return A plotly violin plot
#'
#' @family sentiment
#' @export
plot_sentiment_violin <- function(sentiment_data,
                                  category_var = "category_var",
                                  title = "Sentiment Score Distribution") {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  if (!category_var %in% names(sentiment_data)) {
    stop("Category variable '", category_var, "' not found in data")
  }

  if (!"sentiment_score" %in% names(sentiment_data)) {
    stop("sentiment_score column not found in data")
  }

  plotly::plot_ly(
    sentiment_data,
    x = as.formula(paste0("~", category_var)),
    y = ~sentiment_score,
    type = "violin",
    color = as.formula(paste0("~", category_var)),
    hovertemplate = "%{x}<br>Score: %{y:.3f}<extra></extra>"
  ) %>%
    plotly::layout(
      title = list(
        text = title,
        font = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
      ),
      xaxis = list(
        title = list(text = category_var),
        tickangle = -45,
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = "Sentiment Score"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Roboto, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
      hoverlabel = list(
        align = "left",
        font = list(size = 16, family = "Roboto, sans-serif"),
        maxwidth = 300
      ),
      legend = list(
        font = list(size = 16, family = "Roboto, sans-serif")
      ),
      showlegend = FALSE,
      margin = list(l = 80, r = 40, t = 80, b = 120)
    ) %>%
    plotly::config(displayModeBar = TRUE)
}
