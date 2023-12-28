# Launch TextAnalysisR app ----

#' @title Launch and browse the TextAnalysisR app
#'
#' @name TextAnalysisR.app
#'
#' @description
#' Launch and browse the TextAnalysisR app.
#'
#' @examples
#' if (interactive()) {
#'   library(TextAnalysisR)
#'   TextAnalysisR.app()
#' }
#'
#' @export
#'
#' @import dplyr
#' @import ggplot2
#' @import ggraph
#' @import shiny
#' @import stm
#' @import tidyr
#' @import tidytext
#' @import widyr
#' @importFrom magrittr %>%

TextAnalysisR.app <- function() {
  # Paths to the UI and server files within the TextAnalysisR package
  uiDir <- system.file("TextAnalysisR.app", "ui.R", package = "TextAnalysisR")
  serveDir <- system.file("TextAnalysisR.app", "server.R", package = "TextAnalysisR")

  # Check if the files exist
  if (uiDir == "" || serveDir == "") {
    stop("Error: ui.R or server.R file not found in the TextAnalysisR package.")
  }

  # Source the UI and server files
  source(uiDir, local = TRUE, chdir = TRUE)
  source(serveDir, local = TRUE, chdir = TRUE)

  # Running a Shiny app object
  app <- shinyApp(ui, server)
  runApp(app, display.mode = "normal", launch.browser = TRUE)
}



