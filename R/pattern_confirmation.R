#' @keywords internal
.stratified_folds <- function(y, folds, seed) {
  fold_id <- integer(length(y))
  withr::with_seed(seed, {
    for (lvl in unique(y)) {
      idx <- which(y == lvl)
      fold_id[sample(idx)] <- rep_len(seq_len(folds), length(idx))
    }
  })
  return(fold_id)
}

#' @keywords internal
.downsample <- function(idx, y, seed) {
  counts <- table(y[idx])
  target <- min(counts)
  withr::with_seed(seed, {
    kept <- unlist(lapply(names(counts), function(lvl) {
      pool <- idx[y[idx] == lvl]
      sample(pool, target)
    }), use.names = FALSE)
  })
  return(sort(kept))
}

#' @keywords internal
.class_metrics <- function(actual, predicted, levels_use) {
  rows <- lapply(levels_use, function(lvl) {
    tp <- sum(predicted == lvl & actual == lvl)
    fp <- sum(predicted == lvl & actual != lvl)
    fn <- sum(predicted != lvl & actual == lvl)
    precision <- if (tp + fp == 0) NA_real_ else tp / (tp + fp)
    recall <- if (tp + fn == 0) NA_real_ else tp / (tp + fn)
    f1 <- if (is.na(precision) || is.na(recall) || precision + recall == 0) {
      NA_real_
    } else {
      2 * precision * recall / (precision + recall)
    }
    tibble::tibble(category = lvl, support = sum(actual == lvl),
                   precision = precision, recall = recall, f1 = f1)
  })
  return(dplyr::bind_rows(rows))
}

#' @keywords internal
.fit_predict <- function(x_train, y_train, x_test, method, k) {
  if (method == "knn") {
    if (!requireNamespace("class", quietly = TRUE)) {
      stop("Package 'class' is required for method = 'knn'.", call. = FALSE)
    }
    k_use <- max(1L, min(as.integer(k), length(y_train) - 1L))
    return(as.character(class::knn(x_train, x_test, cl = factor(y_train), k = k_use)))
  }
  if (!requireNamespace("nnet", quietly = TRUE)) {
    stop("Package 'nnet' is required for method = 'multinom'.", call. = FALSE)
  }
  df_train <- as.data.frame(x_train)
  df_train$.y <- factor(y_train)
  fit <- nnet::multinom(.y ~ ., data = df_train, trace = FALSE, MaxNWts = 1e6)
  return(as.character(stats::predict(fit, newdata = as.data.frame(x_test))))
}

#' @title Confirm Categories With Supervised Learning
#'
#' @description
#' Tests whether categories are recoverable from the text representation alone.
#' A classifier maps document embeddings to category labels under stratified
#' cross-validation. Categories a classifier cannot recover are candidates for
#' merging or revision.
#'
#' @param embeddings Numeric matrix or data frame, one row per document.
#' @param categories Category labels, one per row. `NA` and `-1` are excluded;
#'   place them first with [assign_noise()].
#' @param method "knn" (default) or "multinom".
#' @param folds Cross-validation folds (default 5), stratified by category.
#' @param k Neighbours for `method = "knn"` (default 5).
#' @param balance "none" (default) or "downsample" to equalize category sizes
#'   within each training fold.
#' @param seed Random seed for fold assignment and downsampling.
#'
#' @return A list with `overall` (accuracy, macro and weighted F1, counts),
#'   `by_category` (support, precision, recall, F1), `confusion` (a table of
#'   actual against predicted), and `predictions` (per-document actual and
#'   predicted labels).
#'
#' @seealso [assign_noise()] to place unassigned documents before confirming;
#'   [align_categories()] to compare human codes against machine clusters;
#'   [fit_embedding_model()] to produce the categories being tested.
#' @concept topic-modeling
#' @export
validate_categories <- function(embeddings, categories,
                                method = c("knn", "multinom"),
                                folds = 5, k = 5,
                                balance = c("none", "downsample"),
                                seed = 123) {
  method <- match.arg(method)
  balance <- match.arg(balance)
  x <- as.matrix(embeddings)
  if (!is.numeric(x)) stop("embeddings must be numeric.", call. = FALSE)
  y <- as.character(categories)
  if (length(y) != nrow(x)) {
    stop("categories must have one value per row of embeddings.", call. = FALSE)
  }

  keep <- !is.na(y) & y != "-1"
  x <- x[keep, , drop = FALSE]
  y <- y[keep]
  n_excluded <- sum(!keep)

  # a category smaller than the fold count cannot appear in every fold
  counts <- table(y)
  too_small <- names(counts)[counts < folds]
  if (length(too_small) > 0) {
    warning("Dropping category(ies) with fewer members than folds: ",
            paste(too_small, collapse = ", "), call. = FALSE)
    drop <- y %in% too_small
    x <- x[!drop, , drop = FALSE]
    y <- y[!drop]
  }
  levels_use <- sort(unique(y))
  if (length(levels_use) < 2) {
    stop("At least two categories with enough members are required.", call. = FALSE)
  }

  fold_id <- .stratified_folds(y, folds, seed)
  predicted <- character(length(y))
  for (i in seq_len(folds)) {
    test <- which(fold_id == i)
    train <- which(fold_id != i)
    if (balance == "downsample") train <- .downsample(train, y, seed + i)
    predicted[test] <- .fit_predict(x[train, , drop = FALSE], y[train],
                                    x[test, , drop = FALSE], method, k)
  }

  by_category <- .class_metrics(y, predicted, levels_use)
  accuracy <- mean(predicted == y)
  weights <- by_category$support / sum(by_category$support)
  overall <- tibble::tibble(
    accuracy = accuracy,
    macro_f1 = mean(by_category$f1, na.rm = TRUE),
    weighted_f1 = sum(by_category$f1 * weights, na.rm = TRUE),
    n_documents = length(y),
    n_categories = length(levels_use),
    n_excluded = n_excluded,
    method = method,
    folds = folds,
    balance = balance)

  return(list(
    overall = overall,
    by_category = by_category,
    confusion = table(actual = y, predicted = predicted),
    predictions = tibble::tibble(actual = y, predicted = predicted)))
}

#' @title Assign Unclustered Documents to Confirmed Categories
#'
#' @description
#' Places documents left unassigned by density-based clustering, by similarity
#' to confirmed category members. Distance is returned so weak placements stay
#' visible, and `max_distance` leaves distant documents unassigned.
#'
#' @param embeddings Numeric matrix or data frame, one row per document.
#' @param categories Category labels; `NA` or `-1` marks a document unassigned.
#' @param k Neighbours to consult (default 5).
#' @param max_distance Cosine distance beyond which a document stays
#'   unassigned. `NULL` (default) places every document.
#'
#' @return A tibble with `index`, `assigned`, and `distance`, one row per
#'   previously unassigned document.
#'
#' @seealso [validate_categories()] to test the categories afterwards.
#' @concept topic-modeling
#' @export
assign_noise <- function(embeddings, categories, k = 5, max_distance = NULL) {
  x <- as.matrix(embeddings)
  y <- as.character(categories)
  if (length(y) != nrow(x)) {
    stop("categories must have one value per row of embeddings.", call. = FALSE)
  }
  unassigned <- which(is.na(y) | y == "-1")
  empty <- tibble::tibble(index = integer(0), assigned = character(0),
                          distance = numeric(0))
  if (length(unassigned) == 0) return(empty)
  known <- setdiff(seq_along(y), unassigned)
  if (length(known) == 0) {
    stop("No confirmed categories to assign to.", call. = FALSE)
  }

  norm <- sqrt(rowSums(x^2))
  norm[norm == 0] <- 1
  xn <- x / norm
  sim <- xn[unassigned, , drop = FALSE] %*% t(xn[known, , drop = FALSE])

  rows <- lapply(seq_along(unassigned), function(i) {
    ord <- order(sim[i, ], decreasing = TRUE)[seq_len(min(k, length(known)))]
    labs <- y[known][ord]
    best <- names(sort(table(labs), decreasing = TRUE))[1]
    d <- 1 - mean(sim[i, ord][labs == best])
    tibble::tibble(index = unassigned[i], assigned = best, distance = d)
  })
  out <- dplyr::bind_rows(rows)
  if (!is.null(max_distance)) {
    out$assigned[out$distance > max_distance] <- NA_character_
  }
  return(out)
}

#' @title Compare Human Codes With Machine Categories
#'
#' @description
#' Cross-tabulates human against model categories over the same documents and
#' reports the adjusted Rand index, which corrects for chance and does not
#' require the two schemes to share labels or count.
#'
#' @param human Vector of human-assigned categories.
#' @param machine Vector of model-assigned categories, same length and order.
#'
#' @return A list with `crosstab` (human against machine), `adjusted_rand`,
#'   `best_match` (each human category paired with the machine category it
#'   overlaps most, and the share of the human category that pairing covers),
#'   and `n` (documents compared, excluding missing values).
#'
#' @seealso [validate_categories()] for supervised confirmation.
#' @concept topic-modeling
#' @export
align_categories <- function(human, machine) {
  h <- as.character(human)
  m <- as.character(machine)
  if (length(h) != length(m)) {
    stop("human and machine must be the same length.", call. = FALSE)
  }
  keep <- !is.na(h) & !is.na(m) & h != "-1" & m != "-1"
  h <- h[keep]
  m <- m[keep]
  if (length(h) == 0) stop("No documents with both labels.", call. = FALSE)

  tab <- table(human = h, machine = m)

  # adjusted Rand index from the contingency table
  choose2 <- function(v) sum(v * (v - 1) / 2)
  sum_ij <- choose2(as.vector(tab))
  sum_i <- choose2(rowSums(tab))
  sum_j <- choose2(colSums(tab))
  n_pairs <- choose2(length(h))
  expected <- sum_i * sum_j / n_pairs
  maximum <- (sum_i + sum_j) / 2
  ari <- if (maximum == expected) NA_real_ else (sum_ij - expected) / (maximum - expected)

  best <- lapply(rownames(tab), function(r) {
    row <- tab[r, ]
    tibble::tibble(human = r, machine = names(which.max(row))[1],
                   n = as.integer(max(row)),
                   share = as.numeric(max(row) / sum(row)))
  })

  return(list(
    crosstab = tab,
    adjusted_rand = ari,
    best_match = dplyr::bind_rows(best),
    n = length(h)))
}
