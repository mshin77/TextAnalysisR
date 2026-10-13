# add_memo

test_that("add_memo starts a table from NULL", {
  m <- add_memo(NULL, "category", "cost", "Overlaps preparation.")
  expect_equal(nrow(m), 1)
  expect_named(m, c("memo_id", "target_type", "target_id", "round", "text", "created", "stage"))
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
  expect_named(out, c("memo_id", "target_type", "target_id", "round", "text", "created", "stage"))
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
  expect_named(r, c("round", "logged", "n_categories", "n_coded", "n_uncoded", "notes", "stage"))
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

test_that("add_memo numbers past the largest existing id after filtering", {
  m <- add_memo(NULL, "unit", "u1", "first")
  m <- add_memo(m, "unit", "u2", "second")
  m <- add_memo(m, "unit", "u3", "third")
  kept <- m[m$memo_id != "m2", ]
  out <- add_memo(kept, "unit", "u4", "fourth")
  expect_false(any(duplicated(out$memo_id)))
  expect_equal(out$memo_id[nrow(out)], "m4")
})

test_that("code_saturation counts distinct codes in coding order", {
  t0 <- as.POSIXct("2026-01-01 10:00:00", tz = "UTC")
  a <- tibble::tibble(
    unit_id = c("u1", "u2", "u3", "u3", "u4", "u5"),
    code = c("a", "a", "b", "a", NA, "c"),
    status = c("human", "human", "human", "human", "human-none", "ai-confirmed"),
    timestamp = t0 + c(1, 2, 3, 3, 4, 5))
  out <- code_saturation(a, window = 2)
  expect_equal(out$curve$step, 1:5)
  expect_equal(out$curve$new_codes, c(1L, 0L, 1L, 0L, 1L))
  expect_equal(out$curve$distinct_codes, c(1L, 1L, 2L, 2L, 3L))
  expect_equal(out$summary$n_codes, 3L)
  expect_equal(out$summary$last_new_step, 5L)
  expect_false(out$summary$saturated)
})

test_that("code_saturation flags a flat tail and ignores other statuses", {
  t0 <- as.POSIXct("2026-01-01", tz = "UTC")
  a <- tibble::tibble(unit_id = paste0("u", 1:6), code = c("a", "b", "a", "b", "a", "b"),
                      status = c(rep("human", 5), "pending"), timestamp = t0 + 1:6)
  s <- code_saturation(a, window = 3)$summary
  expect_equal(s$n_units, 5L)
  expect_equal(s$last_new_step, 2L)
  expect_equal(s$units_since_new, 3L)
  expect_true(s$saturated)
})

test_that("code_saturation handles no coded units and bad input", {
  e <- code_saturation(tibble::tibble(unit_id = character(0), code = character(0),
                                      status = character(0), timestamp = as.POSIXct(character(0))))
  expect_equal(nrow(e$curve), 0)
  expect_false(e$summary$saturated)
  expect_error(code_saturation(data.frame(x = 1)), "assignments must have")
  expect_error(code_saturation(tibble::tibble(unit_id = "u", code = "a", status = "human",
                                              timestamp = Sys.time()), window = 0), "window")
})
