#' @keywords internal
.ruling_columns <- function() {
  return(tibble::tibble(
    round = integer(0),
    group = character(0),
    n = integer(0),
    verdict = character(0),
    merge_with = character(0),
    label = character(0),
    definition = character(0),
    coder = character(0),
    timestamp = as.POSIXct(character(0))))
}

#' @keywords internal
.read_columns <- function() {
  return(tibble::tibble(
    unit_id = character(0),
    round = integer(0),
    step = character(0),
    coder = character(0),
    timestamp = as.POSIXct(character(0))))
}

#' @keywords internal
.unit_norm <- function(embeddings) {
  x <- as.matrix(embeddings)
  return(x / pmax(sqrt(rowSums(x^2)), .Machine$double.eps))
}

#' @keywords internal
.ruling_ids <- function(embeddings, unit_ids) {
  ids <- if (!is.null(unit_ids)) unit_ids else rownames(embeddings)
  if (is.null(ids)) ids <- as.character(seq_len(nrow(embeddings)))
  return(as.character(ids))
}

#' @keywords internal
.cosine_silhouette <- function(x, parts) {
  d <- 1 - tcrossprod(x)
  s <- vapply(seq_len(nrow(x)), function(i) {
    own <- parts == parts[i]
    a <- if (sum(own) > 1) sum(d[i, own]) / (sum(own) - 1) else 0
    b <- min(vapply(setdiff(unique(parts), parts[i]),
                    function(k) mean(d[i, parts == k]), numeric(1)))
    if (max(a, b) == 0) 0 else (b - a) / max(a, b)
  }, numeric(1))
  return(mean(s))
}

#' @title Evidence for Ruling on Each Group
#'
#' @description
#' Summarizes each group from an unsupervised grouping for a researcher to rule
#' on: how tightly its units sit around their center (coherence), which other
#' group sits closest and how close (separation), units drawn at random inside
#' the group, its furthest member, and the closest group's units that fit this
#' group best. Random draws rather than the most central units are shown,
#' because the center is a biased sample of a group.
#'
#' @param embeddings Numeric matrix, one row per unit.
#' @param groups Group label per unit. `NA`, `0`, and `"0"` mark units no group
#'   holds and are left out.
#' @param unit_ids Unit identifiers. Defaults to `rownames(embeddings)`, then
#'   row positions.
#' @param n_random Units drawn at random inside each group.
#' @param n_neighbour Units shown from the closest group.
#' @param hide Unit identifiers never shown as evidence, such as a drawn
#'   holdout. They still count toward coherence and separation.
#' @param seed Seed for the random draws.
#'
#' @return A tibble with one row per group: `group`, `n`, `coherence`,
#'   `nearest`, `separation`, and list columns `random_units`,
#'   `boundary_unit`, and `neighbour_units` holding unit identifiers.
#'   Separation above coherence raises the question of a merge; the units
#'   answer it.
#'
#' @seealso [apply_ruling()] to act on the verdicts.
#' @concept qualitative-coding
#' @export
group_evidence <- function(embeddings, groups, unit_ids = NULL,
                           n_random = 3, n_neighbour = 2, hide = NULL, seed = 2026) {
  x <- .unit_norm(embeddings)
  ids <- .ruling_ids(embeddings, unit_ids)
  g <- as.character(groups)
  if (length(g) != nrow(x) || length(ids) != nrow(x)) {
    stop("groups and unit_ids must have one entry per row of embeddings.", call. = FALSE)
  }
  held <- !is.na(g) & g != "0"
  keys <- sort(unique(g[held]))
  if (length(keys) < 2) stop("At least two groups are required.", call. = FALSE)
  centers <- t(vapply(keys, function(k) colMeans(x[held & g == k, , drop = FALSE]), numeric(ncol(x))))
  centers <- .unit_norm(centers)
  between <- tcrossprod(centers)
  diag(between) <- -Inf
  showable <- !ids %in% as.character(hide)
  rows <- withr::with_seed(seed, lapply(seq_along(keys), function(j) {
    members <- which(held & g == keys[j])
    fit <- as.vector(x[members, , drop = FALSE] %*% centers[j, ])
    near <- which.max(between[j, ])
    rivals <- which(held & g == keys[near])
    rival_fit <- as.vector(x[rivals, , drop = FALSE] %*% centers[j, ])
    open <- showable[members]
    open_rivals <- rivals[showable[rivals]]
    open_rival_fit <- rival_fit[showable[rivals]]
    pool <- members[open]
    tibble::tibble(
      group = keys[j],
      n = length(members),
      coherence = mean(fit),
      nearest = keys[near],
      separation = between[j, near],
      random_units = list(ids[pool[sample.int(length(pool), min(n_random, length(pool)))]]),
      boundary_unit = list(ids[pool[which.min(fit[open])]]),
      neighbour_units = list(ids[open_rivals[order(open_rival_fit, decreasing = TRUE)][seq_len(min(n_neighbour, length(open_rivals)))]]))
  }))
  return(dplyr::arrange(dplyr::bind_rows(rows), dplyr::desc(.data$n)))
}

#' @title Apply Keep, Split, and Merge Verdicts to a Grouping
#'
#' @description
#' Rebuilds the grouping from one round of researcher verdicts. Merges are
#' applied first, because a merge changes what the remaining groups are
#' compared against; each split then re-partitions that group alone, choosing
#' the number of parts with the highest cosine silhouette among those whose
#' every part meets `min_size`.
#'
#' @param groups Group label per unit, as given to [group_evidence()].
#' @param ruling Data frame with `group`, `verdict` ("keep", "split", or
#'   "merge"), and `merge_with` (the target group, for merges).
#' @param embeddings Numeric matrix, one row per unit; needed for splits.
#' @param min_size Smallest part a split may create.
#' @param max_parts Most parts a split may create.
#' @param seed Seed for the split partition.
#'
#' @return A character vector of new group labels, one per unit. Split parts
#'   take the parent label plus a letter ("g1a", "g1b"). A group the rule cannot
#'   split into parts of at least `min_size` keeps its label, with a warning.
#'
#' @seealso [group_evidence()], [ruling_settled()].
#' @concept qualitative-coding
#' @export
apply_ruling <- function(groups, ruling, embeddings = NULL, min_size = 10,
                         max_parts = 4, seed = 2026) {
  need <- c("group", "verdict")
  miss <- setdiff(need, names(ruling))
  if (length(miss) > 0) {
    stop("ruling is missing column(s): ", paste(miss, collapse = ", "), call. = FALSE)
  }
  verdict <- tolower(trimws(as.character(ruling$verdict)))
  bad <- setdiff(verdict, c("keep", "split", "merge"))
  if (length(bad) > 0) stop("Unknown verdict(s): ", paste(bad, collapse = ", "), call. = FALSE)
  g <- as.character(groups)
  target <- if ("merge_with" %in% names(ruling)) as.character(ruling$merge_with) else rep(NA_character_, nrow(ruling))
  merges <- verdict == "merge"
  if (any(merges & (is.na(target) | !target %in% g))) {
    stop("Every merge needs a merge_with naming an existing group.", call. = FALSE)
  }
  keys <- unique(stats::na.omit(c(g, as.character(ruling$group))))
  parent <- stats::setNames(keys, keys)
  find <- function(k) {
    while (parent[[k]] != k) k <- parent[[k]]
    return(k)
  }
  for (i in which(merges)) {
    from <- find(as.character(ruling$group[i]))
    to <- find(target[i])
    if (from != to) parent[[from]] <- to
  }
  g <- vapply(g, function(k) if (is.na(k)) NA_character_ else find(k), character(1), USE.NAMES = FALSE)
  splits <- intersect(as.character(ruling$group[verdict == "split"]), g)
  if (length(splits) > 0 && is.null(embeddings)) {
    stop("embeddings are required to split a group.", call. = FALSE)
  }
  x <- if (length(splits) > 0) .unit_norm(embeddings) else NULL
  for (k in splits) {
    members <- which(g == k)
    distinct <- nrow(unique(round(x[members, , drop = FALSE], 10)))
    options <- if (length(members) < 2 * min_size || distinct < 2) list() else
      Filter(function(p) !is.null(p) && min(table(p)) >= min_size, withr::with_seed(seed, lapply(
        2:max(2, min(max_parts, floor(length(members) / min_size), distinct)),
        function(parts) tryCatch(stats::kmeans(x[members, , drop = FALSE], centers = parts, nstart = 10)$cluster,
                                 error = function(e) NULL))))
    if (length(options) == 0) {
      warning("Group ", k, " cannot be split into parts of at least ", min_size,
              " units; it keeps its label.", call. = FALSE)
      next
    }
    scores <- vapply(options, function(p) .cosine_silhouette(x[members, , drop = FALSE], p), numeric(1))
    best <- options[[which.max(scores)]]
    g[members] <- paste0(k, letters[best])
  }
  return(g)
}

#' @title Has the Refinement Settled?
#'
#' @description
#' The stopping rule for refinement rounds, stated before the rounds begin: the
#' loop ends on the first round in which every group is kept.
#'
#' @param rulings Verdict table with `round` and `verdict`, as stored in a
#'   coding project.
#'
#' @return `TRUE` when the latest round keeps every group, otherwise `FALSE`.
#'
#' @seealso [apply_ruling()].
#' @concept qualitative-coding
#' @export
ruling_settled <- function(rulings) {
  if (is.null(rulings) || nrow(rulings) == 0) return(FALSE)
  last <- rulings[rulings$round == max(rulings$round), , drop = FALSE]
  return(all(tolower(last$verdict) == "keep"))
}

#' @title Record Units Read During Refinement
#'
#' @description
#' Appends the units a researcher opened, so the report can state how much of
#' the corpus refinement read and the holdout can exclude every unit seen.
#'
#' @param units_read Existing table, or `NULL` to start one.
#' @param unit_ids Units opened.
#' @param round Refinement round.
#' @param step Where they were read, e.g. "ruling" or "annotate".
#' @param coder Coder name.
#'
#' @return The table with new rows appended; units already recorded for the
#'   same round, step, and coder are not repeated.
#'
#' @seealso [draw_holdout()].
#' @concept qualitative-coding
#' @export
log_units_read <- function(units_read = NULL, unit_ids, round, step, coder) {
  if (is.null(units_read)) units_read <- .read_columns()
  rows <- tibble::tibble(unit_id = as.character(unit_ids), round = as.integer(round),
                         step = as.character(step), coder = as.character(coder),
                         timestamp = Sys.time())
  out <- dplyr::bind_rows(units_read, rows)
  return(out[!duplicated(out[c("unit_id", "round", "step", "coder")]), ])
}

#' @title Draw a Blind Holdout From Units Never Read
#'
#' @description
#' Draws the confirmation sample only from units no one opened during
#' refinement, so confirmation tests the codebook on units it was not built
#' from.
#'
#' @param unit_ids All unit identifiers.
#' @param units_read Table from [log_units_read()], or a character vector of
#'   unit identifiers to exclude.
#' @param n Sample size.
#' @param strata Optional grouping, one value per `unit_ids` (e.g. survey
#'   question). Sizes follow each stratum's share of the available units,
#'   rounded by largest remainder. Missing values form their own stratum.
#' @param seed Seed for the draw.
#'
#' @return A character vector of unit identifiers. When fewer than `n` units
#'   remain, all of them are returned with a warning.
#'
#' @seealso [log_units_read()], [code_agreement()].
#' @concept qualitative-coding
#' @export
draw_holdout <- function(unit_ids, units_read = NULL, n = 50, strata = NULL, seed = 2026) {
  ids <- as.character(unit_ids)
  seen <- if (is.data.frame(units_read)) units_read$unit_id else as.character(units_read)
  open <- !ids %in% seen
  if (sum(open) < n) {
    warning(sum(open), " units remain unread; the holdout takes all of them.", call. = FALSE)
    return(ids[open])
  }
  if (is.null(strata)) return(withr::with_seed(seed, sample(ids[open], n)))
  if (length(strata) != length(ids)) {
    stop("strata must have one value per unit_ids.", call. = FALSE)
  }
  s <- as.character(strata)[open]
  s[is.na(s)] <- "(missing)"
  pool <- split(ids[open], s)
  share <- lengths(pool) / sum(open) * n
  size <- floor(share)
  extra <- n - sum(size)
  if (extra > 0) {
    top <- order(share - size, decreasing = TRUE)[seq_len(extra)]
    size[top] <- size[top] + 1
  }
  picks <- withr::with_seed(seed, Map(function(p, k) p[sample.int(length(p), k)], pool, size))
  return(unname(unlist(picks)))
}
