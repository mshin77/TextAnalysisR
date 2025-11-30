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
        font = list(size = 18, color = "#0c1f4a", family = "Montserrat")
      ),
      xaxis = list(
        title = list(text = score_label),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Montserrat, sans-serif")
      ),
      yaxis = list(
        title = list(text = ""),
        categoryorder = "trace",
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Montserrat, sans-serif")
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
        font = list(size = 18, color = "#0c1f4a", family = "Montserrat")
      ),
      xaxis = list(
        title = list(text = "Keyness Score (G\u00b2)"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Montserrat, sans-serif")
      ),
      yaxis = list(
        title = list(text = ""),
        categoryorder = "trace",
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Montserrat, sans-serif")
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
        font = list(size = 18, color = "#0c1f4a", family = "Montserrat")
      ),
      xaxis = list(
        title = list(text = "Keywords"),
        tickangle = -45,
        titlefont = list(size = 16, color = "#0c1f4a", family = "Montserrat, sans-serif"),
        tickfont = list(size = 16, color = "#3B3B3B", family = "Roboto, sans-serif")
      ),
      yaxis = list(
        title = list(text = "Score"),
        titlefont = list(size = 16, color = "#0c1f4a", family = "Montserrat, sans-serif"),
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
