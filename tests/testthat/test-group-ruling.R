.ruling_fixture <- function() {
  set.seed(1)
  centres <- rbind(c(1, 0, 0), c(0, 1, 0), c(0, 0, 1), c(1, 1, 0))
  emb <- do.call(rbind, lapply(1:4, function(k) {
    sweep(matrix(rnorm(30 * 3, sd = 0.05), ncol = 3), 2, centres[k, ], "+")
  }))
  rownames(emb) <- sprintf("u%03d", seq_len(nrow(emb)))
  groups <- c(rep("g1", 30), rep("g2", 30), rep("g3", 30), rep("g1", 30))
  list(emb = emb, groups = groups)
}

test_that("group_evidence returns one row per group with the evidence columns", {
  f <- .ruling_fixture()
  ev <- group_evidence(f$emb, f$groups, n_random = 3, n_neighbour = 2)
  expect_equal(nrow(ev), 3)
  expect_equal(ev$group[1], "g1")
  expect_equal(ev$n, c(60L, 30L, 30L))
  expect_true(all(lengths(ev$random_units) == 3))
  expect_true(all(lengths(ev$neighbour_units) == 2))
  expect_true(all(unlist(ev$random_units[ev$group == "g2"]) %in% rownames(f$emb)[61:90 - 30]))
})

test_that("group_evidence draws the same units for the same seed", {
  f <- .ruling_fixture()
  a <- group_evidence(f$emb, f$groups, seed = 7)
  b <- group_evidence(f$emb, f$groups, seed = 7)
  expect_identical(a$random_units, b$random_units)
})

test_that("group_evidence leaves out units no group holds", {
  f <- .ruling_fixture()
  g <- f$groups
  g[1:5] <- "0"
  g[6:8] <- NA
  ev <- group_evidence(f$emb, g)
  expect_equal(ev$n[ev$group == "g1"], 52L)
  expect_false(any(c("u001", "u006") %in% unlist(ev$random_units)))
})

test_that("apply_ruling merges into the target and resolves mutual merges to one group", {
  f <- .ruling_fixture()
  one_way <- apply_ruling(f$groups, tibble::tibble(group = "g3", verdict = "merge", merge_with = "g2"))
  expect_setequal(unique(one_way), c("g1", "g2"))
  mutual <- apply_ruling(f$groups, tibble::tibble(group = c("g2", "g3"), verdict = "merge",
                                                  merge_with = c("g3", "g2")))
  expect_equal(length(unique(mutual)), 2)
  expect_true(all(mutual[31:90] == mutual[31]))
})

test_that("apply_ruling splits a group that holds two clusters", {
  f <- .ruling_fixture()
  out <- apply_ruling(f$groups, tibble::tibble(group = "g1", verdict = "split"), f$emb, min_size = 10)
  expect_setequal(unique(out[f$groups == "g1"]), c("g1a", "g1b"))
  expect_equal(length(unique(out[1:30])), 1)
  expect_false(out[1] == out[91])
  expect_equal(out[31:90], f$groups[31:90])
})

test_that("apply_ruling keeps a group it cannot split into large enough parts", {
  f <- .ruling_fixture()
  expect_warning(out <- apply_ruling(f$groups, tibble::tibble(group = "g2", verdict = "split"),
                                     f$emb, min_size = 20), "cannot be split")
  expect_equal(out, f$groups)
})

test_that("apply_ruling rejects unknown verdicts and merges without a target", {
  f <- .ruling_fixture()
  expect_error(apply_ruling(f$groups, tibble::tibble(group = "g1", verdict = "drop")), "Unknown verdict")
  expect_error(apply_ruling(f$groups, tibble::tibble(group = "g1", verdict = "merge", merge_with = NA)),
               "merge_with")
  expect_error(apply_ruling(f$groups, tibble::tibble(group = "g1", verdict = "split")), "embeddings")
})

test_that("ruling_settled is TRUE only when the latest round keeps every group", {
  r <- tibble::tibble(round = c(1L, 1L, 2L, 2L), verdict = c("split", "keep", "keep", "keep"))
  expect_true(ruling_settled(r))
  expect_false(ruling_settled(r[1:2, ]))
  expect_false(ruling_settled(NULL))
})

test_that("log_units_read appends without repeating the same read", {
  r <- log_units_read(NULL, c("u1", "u2"), round = 1, step = "ruling", coder = "A")
  r <- log_units_read(r, c("u2", "u3"), round = 1, step = "ruling", coder = "A")
  expect_equal(r$unit_id, c("u1", "u2", "u3"))
  r <- log_units_read(r, "u1", round = 2, step = "ruling", coder = "A")
  expect_equal(nrow(r), 4)
})

test_that("draw_holdout never returns a unit that was read", {
  ids <- sprintf("u%03d", 1:200)
  read <- log_units_read(NULL, ids[1:60], round = 1, step = "ruling", coder = "A")
  h <- draw_holdout(ids, read, n = 50, seed = 3)
  expect_length(h, 50)
  expect_false(any(h %in% ids[1:60]))
  expect_identical(h, draw_holdout(ids, read$unit_id, n = 50, seed = 3))
})

test_that("draw_holdout allocates strata by their share of unread units", {
  ids <- sprintf("u%03d", 1:100)
  strata <- rep(c("Q7", "Q8", "Q9"), times = c(50, 30, 20))
  h <- draw_holdout(ids, NULL, n = 10, strata = strata, seed = 1)
  expect_length(h, 10)
  expect_equal(as.vector(table(strata[match(h, ids)])), c(5, 3, 2))
})

test_that("draw_holdout returns every unread unit with a warning when too few remain", {
  ids <- sprintf("u%02d", 1:10)
  expect_warning(h <- draw_holdout(ids, ids[1:7], n = 5), "remain unread")
  expect_setequal(h, ids[8:10])
})

test_that("apply_ruling keeps a group too small or too uniform to split", {
  f <- .ruling_fixture()
  g <- f$groups
  g[1] <- "g9"
  expect_warning(out <- apply_ruling(g, tibble::tibble(group = "g9", verdict = "split"), f$emb), "cannot be split")
  expect_equal(out[1], "g9")
  flat <- f$emb
  flat[61:90, ] <- matrix(c(0, 0, 1), nrow = 30, ncol = 3, byrow = TRUE)
  expect_warning(out <- apply_ruling(f$groups, tibble::tibble(group = "g3", verdict = "split"), flat), "cannot be split")
  expect_true(all(out[61:90] == "g3"))
})

test_that("apply_ruling splits a part again in a later round", {
  f <- .ruling_fixture()
  once <- apply_ruling(f$groups, tibble::tibble(group = "g1", verdict = "split"), f$emb, min_size = 10)
  part <- once[1]
  twice <- suppressWarnings(apply_ruling(once, tibble::tibble(group = part, verdict = "split"), f$emb, min_size = 10))
  expect_true(all(startsWith(twice[once == part], part)))
  expect_equal(twice[once != part], once[once != part])
})

test_that("group_evidence never shows hidden units", {
  f <- .ruling_fixture()
  hidden <- rownames(f$emb)[c(1:20, 31:50, 61:80, 91:110)]
  ev <- group_evidence(f$emb, f$groups, hide = hidden)
  shown <- unlist(c(ev$random_units, ev$boundary_unit, ev$neighbour_units))
  expect_false(any(shown %in% hidden))
  expect_equal(ev$n, c(60L, 30L, 30L))
})

test_that("draw_holdout checks strata length and keeps missing strata as their own stratum", {
  ids <- sprintf("u%02d", 1:20)
  read <- log_units_read(NULL, character(0), round = 1L, step = "ruling", coder = "A")
  expect_error(draw_holdout(ids, read, n = 4, strata = rep("a", 5)), "one value per")
  s <- c(rep("a", 10), rep(NA, 10))
  picked <- draw_holdout(ids, read, n = 4, strata = s)
  expect_length(picked, 4)
  expect_equal(sum(picked %in% ids[11:20]), 2)
})
