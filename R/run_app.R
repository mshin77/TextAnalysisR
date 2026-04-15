#' @title Launch the TextAnalysisR app
#'
#' @name run_app
#'
#' @description
#' Launch the TextAnalysisR Shiny application.
#'
#' @param launch.browser Logical. Whether to open the app in a browser.
#'   Defaults to `interactive()`, which is FALSE in non-interactive sessions
#'   (e.g., Docker containers, servers).
#'
#' @return No return value, called for side effects (launching Shiny app)
#'
#' @examples
#' if (interactive()) {
#'   library(TextAnalysisR)
#'   run_app()
#' }
#'
#' @export
#'
#' @import dplyr
#' @import ggplot2
#' @import shiny
#' @import tidyr
#' @importFrom magrittr %>%

run_app <- function(launch.browser = interactive()) {
  appDir <- system.file("TextAnalysisR.app", package = "TextAnalysisR")

  if (appDir == "") {
    stop("Error: TextAnalysisR.app directory not found.")
  }

  shiny::runApp(appDir, display.mode = "normal", launch.browser = launch.browser)
}
