#' @title Code Saturation Curve
#'
#' @description
#' Counts distinct codes after each unit is coded, in coding order. A curve with
#' no new code in the last `window` units is evidence the codebook stopped
#' learning; a rising curve means new codes still appear.
#'
#' @param assignments Table with `unit_id`, `code`, `status`, and `timestamp`,
#'   such as `project$assignments`. Rows with status "human", "human-none", or
#'   "ai-confirmed" count; "human-none" adds a unit and no code.
#' @param window Units without a new code that mark the curve saturated (default 10).
#'
#' @return A list with `curve` (`step`, `unit_id`, `new_codes`, `distinct_codes`)
#'   and `summary` (`n_units`, `n_codes`, `last_new_step`, `units_since_new`,
#'   `window`, `saturated`).
#'
#' @seealso [round_summary()] for the uncoded share across rounds.
#' @concept qualitative-coding
#' @export
code_saturation <- function(assignments, window = 10) {
  need <- c("unit_id", "code", "status", "timestamp")
  if (!is.data.frame(assignments) || !all(need %in% names(assignments))) {
    stop("assignments must have columns: ", paste(need, collapse = ", "), ".", call. = FALSE)
  }
  if (length(window) != 1 || is.na(window) || window < 1) {
    stop("window must be a single number of at least 1.", call. = FALSE)
  }
  a <- assignments[assignments$status %in% c("human", "human-none", "ai-confirmed"), , drop = FALSE]
  first_time <- tapply(as.numeric(a$timestamp), a$unit_id, min)
  units <- as.character(names(sort(first_time)))
  new_codes <- tabulate(as.integer(tapply(match(a$unit_id, units), a$code, min)), nbins = length(units))
  curve <- tibble::tibble(step = seq_along(units), unit_id = units, new_codes = new_codes,
                          distinct_codes = cumsum(new_codes))
  last_new <- max(0L, which(new_codes > 0))
  summary <- tibble::tibble(
    n_units = length(units), n_codes = sum(new_codes),
    last_new_step = if (last_new > 0) last_new else NA_integer_,
    units_since_new = length(units) - last_new, window = as.integer(window),
    saturated = length(units) >= window && length(units) - last_new >= window)
  return(list(curve = curve, summary = summary))
}
