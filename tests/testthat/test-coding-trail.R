# add_memo

test_that("add_memo starts a table from NULL", {
  m <- add_memo(NULL, "category", "cost", "Overlaps preparation.")
  expect_equal(nrow(m), 1)
  expect_named(m, c("memo_id", "target_type", "target_id", "round", "text", "created"))
  expect_equal(m$target_type, "category")
  expect_equal(m$target_id, "cost")
  expect_equal(m$round, 1L)
})

test_that("add_memo appends and keeps ids unique", {
  m <- add_memo(NULL, "unit", 88, "Reads as access.")
  m <- add_memo(m, "unit", 91, "Scheduling, not devices.", round = 2)
  expect_equal(nrow(m), 2)
  expect_equal(length(unique(m$memo_id)), 2)
  expect_equal(m$round, c(1L, 2L))
})

test_that("add_memo coerces a numeric target to character", {
  m <- add_memo(NULL, "unit", 88, "text")
  expect_type(m$target_id, "character")
  expect_equal(m$target_id, "88")
})

test_that("add_memo trims surrounding whitespace", {
  m <- add_memo(NULL, "unit", 1, "   padded   ")
  expect_equal(m$text, "padded")
})

test_that("add_memo rejects empty or whitespace-only text", {
  expect_error(add_memo(NULL, "unit", 1, ""), "must not be empty")
  expect_error(add_memo(NULL, "unit", 1, "   "), "must not be empty")
})

test_that("add_memo rejects a non-string or missing text", {
  expect_error(add_memo(NULL, "unit", 1, NA_character_), "single string")
  expect_error(add_memo(NULL, "unit", 1, c("a", "b")), "single string")
})

test_that("add_memo rejects a missing target", {
  expect_error(add_memo(NULL, "unit", NA, "text"), "non-missing")
})

test_that("add_memo rejects an unknown target type", {
  expect_error(add_memo(NULL, "sentence", 1, "text"))
})

# get_memos

test_that("get_memos returns an empty table for NULL input", {
  out <- get_memos(NULL)
  expect_equal(nrow(out), 0)
  expect_named(out, c("memo_id", "target_type", "target_id", "round", "text", "created"))
})

test_that("get_memos with no filter returns everything in order", {
  m <- add_memo(NULL, "unit", 1, "first")
  m <- add_memo(m, "category", "cost", "second")
  expect_equal(get_memos(m)$text, c("first", "second"))
})

test_that("get_memos filters by target type, id, and round", {
  m <- add_memo(NULL, "unit", 1, "unit one")
  m <- add_memo(m, "unit", 2, "unit two", round = 2)
  m <- add_memo(m, "category", "cost", "the category")
  expect_equal(nrow(get_memos(m, target_type = "unit")), 2)
  expect_equal(get_memos(m, target_id = 2)$text, "unit two")
  expect_equal(nrow(get_memos(m, round = 2)), 1)
  expect_equal(nrow(get_memos(m, target_type = "unit", round = 1)), 1)
})

test_that("get_memos returns nothing when no memo matches", {
  m <- add_memo(NULL, "unit", 1, "only one")
  expect_equal(nrow(get_memos(m, target_id = 999)), 0)
})

# log_round

test_that("log_round starts a table and numbers the first round", {
  r <- log_round(NULL, n_categories = 6, n_coded = 177, n_uncoded = 37)
  expect_equal(nrow(r), 1)
  expect_equal(r$round, 1L)
  expect_named(r, c("round", "logged", "n_categories", "n_coded", "n_uncoded", "notes"))
})

test_that("log_round increments the round number on its own", {
  r <- log_round(NULL, 6, 177, 37)
  r <- log_round(r, 8, 201, 13, notes = "two categories added")
  expect_equal(r$round, c(1L, 2L))
  expect_equal(r$notes, c(NA_character_, "two categories added"))
})

test_that("log_round stores counts as integers", {
  r <- log_round(NULL, 6, 177, 37)
  expect_type(r$n_categories, "integer")
  expect_type(r$n_coded, "integer")
  expect_type(r$n_uncoded, "integer")
})

test_that("log_round rejects negative or missing counts", {
  expect_error(log_round(NULL, 6, -1, 37), "n_coded")
  expect_error(log_round(NULL, 6, 177, NA), "n_uncoded")
  expect_error(log_round(NULL, "six", 177, 37), "n_categories")
  expect_error(log_round(NULL, 6, c(1, 2), 37), "n_coded")
})

# round_summary

test_that("round_summary returns an empty table for NULL input", {
  out <- round_summary(NULL)
  expect_equal(nrow(out), 0)
  expect_true("pct_uncoded" %in% names(out))
})

test_that("round_summary computes coverage against total units", {
  r <- log_round(NULL, 6, 177, 37)
  out <- round_summary(r)
  expect_equal(out$n_units, 214L)
  expect_equal(out$pct_uncoded, 100 * 37 / 214)
})

test_that("round_summary reports the change against the previous round", {
  r <- log_round(NULL, 6, 177, 37)
  r <- log_round(r, 8, 201, 13)
  out <- round_summary(r)
  expect_true(is.na(out$change[1]))
  expect_equal(out$change[2], out$pct_uncoded[2] - out$pct_uncoded[1])
  expect_lt(out$change[2], 0)
})

test_that("round_summary reports NA coverage for a round with no units", {
  r <- log_round(NULL, 0, 0, 0)
  expect_true(is.na(round_summary(r)$pct_uncoded))
})

test_that("round_summary never exceeds one hundred percent", {
  r <- log_round(NULL, 3, 0, 50)
  expect_equal(round_summary(r)$pct_uncoded, 100)
})
