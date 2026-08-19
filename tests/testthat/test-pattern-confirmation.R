# fixtures

# centers sit far from the origin in different directions, so the categories
# separate under cosine angle as well as Euclidean distance; sd 0.3 at radius
# 10 leaves no ambiguous neighbourhoods and no random tie-breaking
.confirm_fixture <- function(n_per = 12,
                             centers = list(c(10, 0), c(0, 10), c(-10, 0))) {
  x <- withr::with_seed(42, do.call(rbind, lapply(centers, function(center) {
    cbind(stats::rnorm(n_per, center[1], 0.3),
          stats::rnorm(n_per, center[2], 0.3))
  })))
  list(x = x, y = rep(paste0("c", seq_along(centers)), each = n_per))
}

# .stratified_folds

test_that(".stratified_folds puts every category in every fold", {
  y <- rep(c("a", "b"), each = 10)
  fold_id <- TextAnalysisR:::.stratified_folds(y, folds = 5, seed = 1)
  expect_equal(sort(unique(fold_id)), 1:5)
  expect_true(all(table(y, fold_id) == 2))
})

test_that(".stratified_folds spreads an indivisible category across folds", {
  y <- rep("a", 12)
  fold_id <- TextAnalysisR:::.stratified_folds(y, folds = 5, seed = 1)
  expect_equal(sort(as.vector(table(fold_id))), c(2L, 2L, 2L, 3L, 3L))
})

test_that(".stratified_folds is reproducible for a given seed", {
  y <- rep(c("a", "b", "c"), each = 7)
  expect_equal(TextAnalysisR:::.stratified_folds(y, 3, seed = 99),
               TextAnalysisR:::.stratified_folds(y, 3, seed = 99))
})

# .downsample

test_that(".downsample equalizes categories to the smallest", {
  y <- c(rep("a", 10), rep("b", 3))
  kept <- TextAnalysisR:::.downsample(seq_along(y), y, seed = 1)
  expect_equal(as.vector(table(y[kept])), c(3L, 3L))
  expect_false(is.unsorted(kept))
})

# .class_metrics

test_that(".class_metrics scores perfect prediction as one", {
  y <- c("a", "a", "b", "b")
  m <- TextAnalysisR:::.class_metrics(y, y, c("a", "b"))
  expect_equal(m$precision, c(1, 1))
  expect_equal(m$recall, c(1, 1))
  expect_equal(m$f1, c(1, 1))
  expect_equal(m$support, c(2L, 2L))
})

test_that(".class_metrics returns NA precision for a never-predicted category", {
  actual <- c("a", "a", "b")
  predicted <- c("a", "a", "a")
  m <- TextAnalysisR:::.class_metrics(actual, predicted, c("a", "b"))
  b <- m[m$category == "b", ]
  expect_true(is.na(b$precision))
  expect_equal(b$recall, 0)
  expect_true(is.na(b$f1))
})

# validate_categories

test_that("validate_categories recovers well-separated categories", {
  skip_if_not_installed("class")
  f <- .confirm_fixture()
  res <- validate_categories(f$x, f$y, folds = 4)
  expect_gt(res$overall$accuracy, 0.95)
  expect_gt(res$overall$macro_f1, 0.95)
})

test_that("validate_categories returns the documented structure", {
  skip_if_not_installed("class")
  f <- .confirm_fixture()
  res <- validate_categories(f$x, f$y, folds = 4)
  expect_named(res, c("overall", "by_category", "confusion", "predictions"))
  expect_named(res$by_category,
               c("category", "support", "precision", "recall", "f1"))
  expect_equal(nrow(res$by_category), 3)
  expect_equal(nrow(res$predictions), length(f$y))
  expect_equal(dim(res$confusion), c(3L, 3L))
  expect_equal(res$overall$n_documents, length(f$y))
  expect_equal(res$overall$n_categories, 3)
  expect_equal(sum(res$by_category$support), length(f$y))
})

test_that("validate_categories excludes NA and the unassigned label", {
  skip_if_not_installed("class")
  f <- .confirm_fixture()
  y <- f$y
  y[1:3] <- NA
  y[13:14] <- "0"
  res <- validate_categories(f$x, y, folds = 4)
  expect_equal(res$overall$n_excluded, 5)
  expect_equal(res$overall$n_documents, length(y) - 5)
  expect_false(any(c("0", NA) %in% res$predictions$actual))
})

test_that("validate_categories drops a category smaller than the fold count", {
  skip_if_not_installed("class")
  f <- .confirm_fixture()
  y <- f$y
  y[25:33] <- "c2"
  expect_warning(res <- validate_categories(f$x, y, folds = 5), "Dropping")
  expect_equal(res$overall$n_categories, 2)
  expect_false("c3" %in% res$by_category$category)
})

test_that("validate_categories errors when fewer than two categories survive", {
  skip_if_not_installed("class")
  f <- .confirm_fixture(centers = list(c(10, 0), c(0, 10)))
  y <- f$y
  y[13:22] <- "c1"
  expect_error(suppressWarnings(validate_categories(f$x, y, folds = 5)),
               "At least two categories")
})

test_that("validate_categories errors on a length mismatch", {
  f <- .confirm_fixture()
  expect_error(validate_categories(f$x, f$y[-1]), "one value per row")
})

test_that("validate_categories errors on non-numeric embeddings", {
  expect_error(
    validate_categories(matrix(letters[1:8], ncol = 2), rep(c("a", "b"), 2)),
    "numeric"
  )
})

test_that("validate_categories is reproducible for a given seed", {
  skip_if_not_installed("class")
  f <- .confirm_fixture()
  a <- validate_categories(f$x, f$y, folds = 4, seed = 7)
  b <- validate_categories(f$x, f$y, folds = 4, seed = 7)
  expect_equal(a$predictions, b$predictions)
})

test_that("validate_categories records method, folds, and balance", {
  skip_if_not_installed("class")
  f <- .confirm_fixture()
  res <- validate_categories(f$x, f$y, folds = 3, balance = "downsample")
  expect_equal(res$overall$method, "knn")
  expect_equal(res$overall$folds, 3)
  expect_equal(res$overall$balance, "downsample")
})

test_that("validate_categories runs the multinom method", {
  skip_if_not_installed("nnet")
  f <- .confirm_fixture()
  res <- validate_categories(f$x, f$y, method = "multinom", folds = 3)
  expect_equal(res$overall$method, "multinom")
  expect_gt(res$overall$accuracy, 0.95)
})

test_that("validate_categories rejects an unknown method", {
  f <- .confirm_fixture()
  expect_error(validate_categories(f$x, f$y, method = "randomforest"))
})

# assign_noise

test_that("assign_noise returns an empty tibble when nothing is unassigned", {
  f <- .confirm_fixture()
  out <- assign_noise(f$x, f$y)
  expect_equal(nrow(out), 0)
  expect_named(out, c("index", "assigned", "distance"))
})

test_that("assign_noise places NA rows with their nearest category", {
  f <- .confirm_fixture()
  y <- f$y
  y[c(1, 13, 25)] <- NA
  out <- assign_noise(f$x, y, k = 5)
  expect_equal(out$index, c(1L, 13L, 25L))
  expect_equal(out$assigned, c("c1", "c2", "c3"))
  expect_true(all(out$distance >= 0))
})

test_that("assign_noise treats the unassigned label as not yet placed", {
  f <- .confirm_fixture()
  y <- f$y
  y[13] <- "0"
  out <- assign_noise(f$x, y)
  expect_equal(out$index, 13L)
  expect_equal(out$assigned, "c2")
})

test_that("assign_noise leaves distant documents unassigned under max_distance", {
  f <- .confirm_fixture()
  y <- f$y
  y[1] <- NA
  near <- assign_noise(f$x, y, max_distance = 1)
  far <- assign_noise(f$x, y, max_distance = 0)
  expect_false(is.na(near$assigned))
  expect_true(is.na(far$assigned))
  expect_equal(far$distance, near$distance)
})

test_that("assign_noise errors when no category is left to assign to", {
  f <- .confirm_fixture()
  expect_error(assign_noise(f$x, rep(NA_character_, length(f$y))),
               "No confirmed categories")
})

test_that("assign_noise errors on a length mismatch", {
  f <- .confirm_fixture()
  expect_error(assign_noise(f$x, f$y[-1]), "one value per row")
})

test_that("assign_noise survives a zero-length embedding row", {
  f <- .confirm_fixture()
  x <- f$x
  x[1, ] <- 0
  y <- f$y
  y[1] <- NA
  out <- assign_noise(x, y)
  expect_equal(nrow(out), 1)
  expect_false(is.na(out$distance))
  expect_false(is.na(out$assigned))
})

# align_categories

test_that("align_categories scores identical partitions as one", {
  g <- rep(c("a", "b", "c"), each = 4)
  res <- align_categories(g, g)
  expect_equal(res$adjusted_rand, 1)
  expect_equal(res$n, 12)
})

test_that("align_categories ignores the labels themselves", {
  human <- rep(c("a", "b", "c"), each = 4)
  machine <- rep(c("3", "1", "2"), each = 4)
  expect_equal(align_categories(human, machine)$adjusted_rand, 1)
})

test_that("align_categories scores an independent partition near zero", {
  human <- rep(c("a", "b"), each = 20)
  machine <- withr::with_seed(3, sample(rep(c("x", "y"), 20)))
  expect_lt(abs(align_categories(human, machine)$adjusted_rand), 0.3)
})

test_that("align_categories reports the dominant overlap per human category", {
  human <- c("a", "a", "a", "b", "b")
  machine <- c("x", "x", "y", "y", "y")
  res <- align_categories(human, machine)
  a <- res$best_match[res$best_match$human == "a", ]
  b <- res$best_match[res$best_match$human == "b", ]
  expect_equal(a$machine, "x")
  expect_equal(a$n, 2L)
  expect_equal(a$share, 2 / 3)
  expect_equal(b$machine, "y")
  expect_equal(b$share, 1)
})

test_that("align_categories excludes rows missing either label", {
  human <- c("a", "a", NA, "b", "b")
  machine <- c("x", "x", "y", "0", "y")
  res <- align_categories(human, machine)
  expect_equal(res$n, 3)
  expect_equal(dim(res$crosstab), c(2L, 2L))
})

test_that("align_categories returns NA when neither partition splits anything", {
  g <- rep("a", 6)
  expect_true(is.na(align_categories(g, rep("x", 6))$adjusted_rand))
})

test_that("align_categories errors on a length mismatch", {
  expect_error(align_categories(c("a", "b"), "x"), "same length")
})

test_that("align_categories errors when no document carries both labels", {
  expect_error(align_categories(c(NA, NA), c("x", "y")), "No documents")
})

# the package's own outlier label

test_that("validate_categories excludes the package's 0 outlier label by default", {
  skip_if_not_installed("class")
  f <- .confirm_fixture()
  y <- f$y
  y[1:5] <- "0"
  res <- validate_categories(f$x, y, folds = 4)
  expect_equal(res$overall$n_excluded, 5)
  expect_false("0" %in% res$by_category$category)
})

test_that("validate_categories honours a different unassigned label", {
  skip_if_not_installed("class")
  f <- .confirm_fixture()
  y <- f$y
  y[1:5] <- "-1"
  res <- validate_categories(f$x, y, folds = 4, unassigned = -1)
  expect_equal(res$overall$n_excluded, 5)
  expect_false("-1" %in% res$by_category$category)
})

test_that("validate_categories keeps 0 when told a different sentinel", {
  skip_if_not_installed("class")
  f <- .confirm_fixture()
  y <- f$y
  y[1:12] <- "0"
  res <- validate_categories(f$x, y, folds = 4, unassigned = -1)
  expect_equal(res$overall$n_excluded, 0)
  expect_true("0" %in% res$by_category$category)
})

test_that("assign_noise places the package's 0 outliers", {
  f <- .confirm_fixture()
  y <- f$y
  y[c(1, 13, 25)] <- "0"
  out <- assign_noise(f$x, y)
  expect_equal(out$index, c(1L, 13L, 25L))
  expect_equal(out$assigned, c("c1", "c2", "c3"))
})

test_that("align_categories drops 0 from either vector by default", {
  human <- c("a", "a", "0", "b", "b")
  machine <- c("x", "x", "y", "0", "y")
  res <- align_categories(human, machine)
  expect_equal(res$n, 3)
})

test_that("discovery output feeds confirmation without a phantom category", {
  skip_if_not_installed("class")
  f <- .confirm_fixture()
  # fit_embedding_model shifts BERTopic labels by one, so outliers arrive as 0
  emitted <- ifelse(seq_along(f$y) %in% 1:20, 0L, match(f$y, sort(unique(f$y))))
  res <- validate_categories(f$x, emitted, folds = 4)
  expect_equal(res$overall$n_excluded, 20)
  expect_equal(res$overall$n_categories, length(setdiff(unique(emitted), 0)))
})
