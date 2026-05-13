test_that("prep_texts works with default settings", {
  df <- data.frame(united_texts = c(
    "Solve 2 + 3 = 5 for the value of x.",
    "Find the area of a circle with radius 4 cm."
  ))
  toks <- prep_texts(df, text_field = "united_texts")
  expect_s3_class(toks, "tokens")
  expect_equal(length(toks), 2)
})

test_that("math_mode = TRUE preserves numbers", {
  df <- data.frame(united_texts = c("Compute 3 + 4 = 7 quickly."))
  toks_default <- prep_texts(df, text_field = "united_texts")
  toks_math    <- prep_texts(df, text_field = "united_texts", math_mode = TRUE)

  flat_default <- as.character(unlist(as.list(toks_default)))
  flat_math    <- as.character(unlist(as.list(toks_math)))

  expect_false(any(c("3", "4", "7") %in% flat_default))
  expect_true(all(c("3", "4", "7") %in% flat_math))
})

test_that("math_mode = TRUE preserves math operators", {
  df <- data.frame(united_texts = c("Solve x + y = 10 and x - y = 2."))
  toks <- prep_texts(df, text_field = "united_texts", math_mode = TRUE)
  flat <- as.character(unlist(as.list(toks)))
  expect_true("+" %in% flat)
  expect_true("=" %in% flat)
  expect_true("-" %in% flat)
})

test_that("math_mode = TRUE strips sentence-end punctuation", {
  df <- data.frame(united_texts = c("Solve 2 + 3, then check the answer."))
  toks <- prep_texts(df, text_field = "united_texts", math_mode = TRUE)
  flat <- as.character(unlist(as.list(toks)))
  expect_false("." %in% flat)
  expect_false("," %in% flat)
})

test_that("math_mode overrides remove_* flags even when set TRUE", {
  df <- data.frame(united_texts = c("3 + 4 = 7"))
  toks <- prep_texts(df, text_field = "united_texts",
                     math_mode = TRUE,
                     remove_punct   = TRUE,
                     remove_symbols = TRUE,
                     remove_numbers = TRUE)
  flat <- as.character(unlist(as.list(toks)))
  expect_true("3" %in% flat)
  expect_true("+" %in% flat)
})
