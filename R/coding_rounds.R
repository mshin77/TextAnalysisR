#' @keywords internal
.round_columns <- function() {
  return(tibble::tibble(
    round = integer(0),
    logged = as.POSIXct(character(0)),
    n_categories = integer(0),
    n_coded = integer(0),
    n_uncoded = integer(0),
    notes = character(0)))
}

#' @keywords internal
.check_count <- function(value, name) {
  if (length(value) != 1 || is.na(value) || !is.numeric(value) || value < 0) {
    stop(name, " must be a single non-negative number.", call. = FALSE)
  }
  return(as.integer(value))
}

#' @title Record a Refinement Round
#'
#' @description
#' Appends one row per pass through discovery, coding, and revision. The round
#' number increments on its own, so repeated calls build the record in order.
#'
#' @param rounds Existing round table, or `NULL` to start one.
#' @param n_categories Categories in the codebook for this round.
#' @param n_coded Units that received at least one code.
#' @param n_uncoded Units that received none, from [uncoded_units()].
#' @param notes What changed this round. Optional.
#'
#' @return The round table with one row appended.
#'
#' @seealso [round_summary()] for the coverage trend across rounds;
#'   [uncoded_units()] for the count this records.
#' @concept qualitative-coding
#' @export
log_round <- function(rounds = NULL, n_categories, n_coded, n_uncoded,
                      notes = NA_character_) {
  if (is.null(rounds)) rounds <- .round_columns()
  row <- tibble::tibble(
    round = nrow(rounds) + 1L,
    logged = Sys.time(),
    n_categories = .check_count(n_categories, "n_categories"),
    n_coded = .check_count(n_coded, "n_coded"),
    n_uncoded = .check_count(n_uncoded, "n_uncoded"),
    notes = as.character(notes))
  return(dplyr::bind_rows(rounds, row))
}

#' @title Coverage Trend Across Refinement Rounds
#'
#' @description
#' Reports what share of units the codebook left untouched in each round, and how
#' that share moved. A falling share is evidence the codebook is still learning
#' from the text; a flat share is evidence it has stopped.
#'
#' @param rounds Round table from [log_round()].
#'
#' @return A tibble with `round`, `n_categories`, `n_units`, `n_coded`,
#'   `n_uncoded`, `pct_uncoded`, and `change` in percentage points against the
#'   previous round (`NA` for the first).
#'
#' @seealso [log_round()] to build the table.
#' @concept qualitative-coding
#' @export
round_summary <- function(rounds) {
  if (is.null(rounds) || nrow(rounds) == 0) {
    return(tibble::tibble(round = integer(0), n_categories = integer(0),
                          n_units = integer(0), n_coded = integer(0),
                          n_uncoded = integer(0), pct_uncoded = numeric(0),
                          change = numeric(0)))
  }
  n_units <- rounds$n_coded + rounds$n_uncoded
  # a round with no units has no coverage to report, rather than a zero
  pct <- ifelse(n_units == 0, NA_real_, 100 * rounds$n_uncoded / n_units)
  return(tibble::tibble(
    round = rounds$round,
    n_categories = rounds$n_categories,
    n_units = n_units,
    n_coded = rounds$n_coded,
    n_uncoded = rounds$n_uncoded,
    pct_uncoded = pct,
    change = c(NA_real_, diff(pct))))
}
