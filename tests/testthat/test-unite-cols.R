test_that("unite_cols pastes selected columns with single-space separator", {
  df <- data.frame(
    title = c("Hello", "Foo"),
    body  = c("World", "Bar"),
    other = c("x", "y"),
    stringsAsFactors = FALSE
  )
  out <- unite_cols(df, listed_vars = c("title", "body"))
  expect_true("united_texts" %in% names(out))
  expect_equal(out$united_texts, c("Hello World", "Foo Bar"))
})

test_that("unite_cols replaces NA cells with empty string", {
  df <- data.frame(
    title = c("Alpha", NA, "Gamma"),
    body  = c(NA, "Beta", "Delta"),
    stringsAsFactors = FALSE
  )
  out <- unite_cols(df, listed_vars = c("title", "body"))
  expect_equal(out$united_texts, c("Alpha", "Beta", "Gamma Delta"))
  expect_false(any(grepl("\\bNA\\b", out$united_texts)))
})

test_that("unite_cols collapses internal whitespace from NA gaps", {
  df <- data.frame(
    a = c("one", NA),
    b = c(NA, "two"),
    c = c("three", "four"),
    stringsAsFactors = FALSE
  )
  out <- unite_cols(df, listed_vars = c("a", "b", "c"))
  expect_equal(out$united_texts, c("one three", "two four"))
})

test_that("unite_cols preserves docvar columns alongside united_texts", {
  df <- data.frame(
    title = c("A", "B"),
    body  = c("a", "b"),
    year  = c(2020L, 2021L),
    group = c("x", "y"),
    stringsAsFactors = FALSE
  )
  out <- unite_cols(df, listed_vars = c("title", "body"))
  expect_true(all(c("year", "group") %in% names(out)))
  expect_equal(out$year, c(2020L, 2021L))
})
