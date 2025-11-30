#' Apply Standard Plotly Layout
#'
#' @description
#' Applies consistent layout styling to plotly plots following TextAnalysisR design standards.
#' This ensures all plots have uniform fonts, colors, margins, and interactive features.
#'
#' @param plot A plotly plot object
#' @param title Plot title text (optional)
#' @param xaxis_title X-axis title (optional)
#' @param yaxis_title Y-axis title (optional)
#' @param margin List of margins: list(t, b, l, r) in pixels (default: list(t = 60, b = 80, l = 80, r = 40))
#' @param show_legend Logical, whether to show legend (default: FALSE)
#'
#' @return A plotly plot object with standardized layout
#'
#' @details
#' Design standards applied:
#' - Title: 20px Roboto, #0c1f4a
#' - Axis titles: 18px Roboto, #0c1f4a
#' - Axis labels: 18px Roboto, #3B3B3B
#' - Hover tooltips: 16px Roboto
#' - WCAG AA compliant colors
#'
#' @export
#'
#' @examples
#' \dontrun{
#' library(plotly)
#' p <- plot_ly(x = 1:10, y = rnorm(10), type = "scatter", mode = "markers")
#' p %>% apply_standard_plotly_layout(
#'   title = "My Plot",
#'   xaxis_title = "X Values",
#'   yaxis_title = "Y Values"
#' )
#' }
apply_standard_plotly_layout <- function(plot,
                                         title = NULL,
                                         xaxis_title = NULL,
                                         yaxis_title = NULL,
                                         margin = list(t = 60, b = 80, l = 80, r = 40),
                                         show_legend = FALSE) {

  if (!requireNamespace("plotly", quietly = TRUE)) {
    stop("Package 'plotly' is required. Please install it.")
  }

  layout_config <- list(
    font = list(family = "Roboto, sans-serif", size = 16, color = "#3B3B3B"),
    hoverlabel = list(
      font = list(size = 16, family = "Roboto, sans-serif"),
      align = "left"
    ),
    margin = margin,
    showlegend = show_legend,
    xaxis = list(
      tickfont = list(size = 18, color = "#3B3B3B", family = "Roboto, sans-serif"),
      titlefont = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
    ),
    yaxis = list(
      tickfont = list(size = 18, color = "#3B3B3B", family = "Roboto, sans-serif"),
      titlefont = list(size = 18, color = "#0c1f4a", family = "Roboto, sans-serif")
    )
  )

  if (!is.null(title)) {
    layout_config$title <- list(
      text = title,
      font = list(size = 20, color = "#0c1f4a", family = "Roboto, sans-serif")
    )
  }

  if (!is.null(xaxis_title)) {
    layout_config$xaxis$title <- list(text = xaxis_title)
  }

  if (!is.null(yaxis_title)) {
    layout_config$yaxis$title <- list(text = yaxis_title)
  }

  plot %>%
    plotly::layout(layout_config) %>%
    plotly::config(displayModeBar = TRUE)
}


#' Get Standard Plotly Hover Label Configuration
#'
#' @description
#' Returns standardized hover label styling for plotly plots.
#'
#' @param bgcolor Background color (default: "#ffffff")
#' @param fontcolor Font color (default: "#0c1f4a")
#'
#' @return A list of hover label configuration parameters
#'
#' @export
#'
#' @examples
#' \dontrun{
#' hover_config <- get_plotly_hover_config()
#' plot_ly(..., hoverlabel = hover_config)
#' }
get_plotly_hover_config <- function(bgcolor = "#ffffff", fontcolor = "#0c1f4a") {
  list(
    bgcolor = bgcolor,
    bordercolor = bgcolor,
    font = list(
      family = "Roboto, sans-serif",
      size = 16,
      color = fontcolor
    ),
    align = "left",
    namelength = -1
  )
}


#' Create Standard ggplot2 Theme
#'
#' @description
#' Returns a standardized ggplot2 theme matching TextAnalysisR design standards.
#'
#' @param base_size Base font size (default: 14)
#'
#' @return A ggplot2 theme object
#'
#' @export
#'
#' @examples
#' \dontrun{
#' library(ggplot2)
#' ggplot(mtcars, aes(mpg, wt)) +
#'   geom_point() +
#'   create_standard_ggplot_theme()
#' }
create_standard_ggplot_theme <- function(base_size = 14) {

  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package 'ggplot2' is required. Please install it.")
  }

  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        size = 20,
        color = "#0c1f4a",
        hjust = 0.5,
        family = "Roboto"
      ),
      axis.title = ggplot2::element_text(
        size = 18,
        color = "#0c1f4a",
        family = "Roboto"
      ),
      axis.text = ggplot2::element_text(
        size = 18,
        color = "#3B3B3B",
        family = "Roboto"
      ),
      strip.text = ggplot2::element_text(
        size = 18,
        color = "#0c1f4a",
        family = "Roboto"
      ),
      legend.text = ggplot2::element_text(
        size = 16,
        color = "#3B3B3B",
        family = "Roboto"
      ),
      legend.title = ggplot2::element_text(
        size = 16,
        color = "#0c1f4a",
        family = "Roboto"
      )
    )
}


#' Get Sentiment Color Palette
#'
#' @description
#' Returns standardized color mapping for sentiment analysis.
#'
#' @return Named vector of colors
#'
#' @export
get_sentiment_colors <- function() {
  c(
    "positive" = "#10B981",
    "negative" = "#EF4444",
    "neutral" = "#6B7280"
  )
}


#' Generate Sentiment Color Gradient
#'
#' @description
#' Generates a color based on sentiment score using a gradient from red (negative)
#' through gray (neutral) to green (positive).
#'
#' @param score Numeric sentiment score (typically -1 to 1)
#'
#' @return Hex color string
#'
#' @export
#'
#' @examples
#' get_sentiment_color(-0.8)  # Red
#' get_sentiment_color(0)     # Gray
#' get_sentiment_color(0.8)   # Green
get_sentiment_color <- function(score) {
  normalized_score <- (score + 1) / 2
  normalized_score <- pmax(0, pmin(1, normalized_score))

  if (normalized_score < 0.5) {
    t <- normalized_score * 2
    r <- round(185 * (1 - t) + 75 * t)
    g <- round(67 * (1 - t) + 181 * t)
    b <- round(68 * (1 - t) + 67 * t)
  } else {
    t <- (normalized_score - 0.5) * 2
    r <- round(75 * (1 - t) + 16 * t)
    g <- round(181 * (1 - t) + 185 * t)
    b <- round(67 * (1 - t) + 129 * t)
  }

  sprintf("#%02X%02X%02X", r, g, b)
}


#' Create Message Data Table
#'
#' @description
#' Creates a formatted DT::datatable displaying an informational message.
#' Useful for showing status messages in place of empty tables.
#'
#' @param message Character string message to display
#' @param font_size Font size (default: "16px")
#' @param color Text color (default: "#6c757d")
#'
#' @return A DT::datatable object
#'
#' @export
#'
#' @examples
#' \dontrun{
#' create_message_table("No data available. Please run analysis first.")
#' }
create_message_table <- function(message,
                                 font_size = "16px",
                                 color = "#6c757d") {

  if (!requireNamespace("DT", quietly = TRUE)) {
    stop("Package 'DT' is required. Please install it.")
  }

  DT::datatable(
    data.frame(Message = message),
    rownames = FALSE,
    options = list(
      dom = "t",
      ordering = FALSE,
      columnDefs = list(
        list(className = 'dt-center', targets = "_all")
      ),
      initComplete = htmlwidgets::JS(
        sprintf(
          "function(settings, json) {
            $(this.api().table().container()).find('td').css({
              'font-size': '%s',
              'color': '%s',
              'padding': '40px',
              'text-align': 'center'
            });
          }",
          font_size,
          color
        )
      )
    ),
    class = 'cell-border stripe'
  )
}
