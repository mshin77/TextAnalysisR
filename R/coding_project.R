#' @keywords internal
.project_version <- 1L

#' @keywords internal
.project_tables <- function() {
  return(list(
    codebook = tibble::tibble(code = character(0), definition = character(0),
                              example = character(0), color = character(0)),
    assignments = tibble::tibble(doc_id = character(0), unit_id = character(0),
                                 start = integer(0), end = integer(0),
                                 code = character(0), coder = character(0),
                                 confidence = numeric(0), rationale = character(0),
                                 status = character(0),
                                 timestamp = as.POSIXct(character(0))),
    memos = .memo_columns(),
    positionality = tibble::tibble(coder = character(0), statement = character(0),
                                   timestamp = as.POSIXct(character(0))),
    rounds = .round_columns(),
    rulings = .ruling_columns(),
    units_read = .read_columns(),
    holdout = tibble::tibble(unit_id = character(0))))
}

#' @title Start a Qualitative Coding Project
#'
#' @description
#' Creates the project record one coder works in: the codebook, code
#' assignments, memos, positionality, the refinement rounds and the verdict on
#' every group, the units read during refinement, and the holdout drawn for
#' confirmation. One coder per project; [merge_codes()] combines coders.
#'
#' @param coder Coder name, recorded on every assignment and verdict.
#' @param context Free-text note on the sample and setting. Optional.
#'
#' @return A list of class `coding_project`.
#'
#' @seealso [save_coding_project()], [read_coding_project()].
#' @concept qualitative-coding
#' @export
new_coding_project <- function(coder, context = NA_character_) {
  if (!is.character(coder) || length(coder) != 1 || is.na(coder) || !nzchar(trimws(coder))) {
    stop("coder must be a single non-empty string.", call. = FALSE)
  }
  project <- c(list(version = .project_version, coder = trimws(coder),
                    context = as.character(context), created = Sys.time(),
                    saved = as.POSIXct(NA)),
               .project_tables())
  return(structure(project, class = "coding_project"))
}

#' @title Save a Qualitative Coding Project
#'
#' @description
#' Writes the project to an `.rds` file. The file is written beside the target
#' and renamed into place, so an interrupted save leaves the previous file
#' intact.
#'
#' @param project Project from [new_coding_project()] or [read_coding_project()].
#' @param path Destination `.rds` path.
#' @param backup Logical; copy the file being replaced to `<path>.bak` first.
#'   An app that saves repeatedly sets this only on its first save, so the
#'   backup keeps the state from before the session.
#'
#' @return The saved project, invisibly, with its `saved` time updated.
#'
#' @seealso [read_coding_project()].
#' @concept qualitative-coding
#' @export
save_coding_project <- function(project, path, backup = TRUE) {
  if (!inherits(project, "coding_project")) {
    stop("project must come from new_coding_project() or read_coding_project().", call. = FALSE)
  }
  project$saved <- Sys.time()
  tmp <- tempfile(pattern = ".coding-", tmpdir = dirname(path), fileext = ".rds")
  tryCatch(saveRDS(unclass(project), tmp), error = function(e) {
    unlink(tmp)
    stop("Could not write ", path, ": ", conditionMessage(e), call. = FALSE)
  })
  if (isTRUE(backup) && file.exists(path)) file.copy(path, paste0(path, ".bak"), overwrite = TRUE)
  moved <- file.rename(tmp, path)
  if (!moved) {
    copied <- file.copy(tmp, path, overwrite = TRUE)
    unlink(tmp)
    if (!copied) stop("Could not write ", path, call. = FALSE)
  }
  return(invisible(project))
}

#' @title Read a Qualitative Coding Project
#'
#' @description
#' Reads a project saved by [save_coding_project()]. Tables and columns added
#' in later versions are filled in empty, so older files open without loss.
#'
#' @param path Path to the `.rds` file.
#'
#' @return A list of class `coding_project`.
#'
#' @seealso [save_coding_project()].
#' @concept qualitative-coding
#' @export
read_coding_project <- function(path) {
  raw <- readRDS(path)
  if (!is.list(raw) || is.null(raw$coder)) {
    stop(path, " is not a coding project file.", call. = FALSE)
  }
  if (isTRUE(raw$version > .project_version)) {
    stop(path, " was saved by a newer version of TextAnalysisR.", call. = FALSE)
  }
  tables <- .project_tables()
  missing_tables <- setdiff(names(tables), names(raw))
  raw[missing_tables] <- tables[missing_tables]
  for (tbl in names(tables)) {
    absent <- setdiff(names(tables[[tbl]]), names(raw[[tbl]]))
    for (col in absent) raw[[tbl]][[col]] <- rep(tables[[tbl]][[col]][NA_integer_], nrow(raw[[tbl]]))
  }
  raw$version <- .project_version
  return(structure(raw, class = "coding_project"))
}

#' @method print coding_project
#' @export
print.coding_project <- function(x, ...) {
  cat(sprintf(paste0("Coding project: %s\n",
                     "  codes %d | assignments %d | rounds %d | verdicts %d | units read %d | holdout %d\n",
                     "  saved %s\n"),
              x$coder, nrow(x$codebook), nrow(x$assignments), length(unique(x$rulings$round)),
              nrow(x$rulings), nrow(x$units_read), nrow(x$holdout),
              if (is.na(x$saved)) "never" else format(x$saved, "%Y-%m-%d %H:%M:%S")))
  return(invisible(x))
}
