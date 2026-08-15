.mm_tokens <- function(x, ...) {
  toks <- prep_texts(tibble::tibble(united_texts = x), text_field = "united_texts",
                     remove_stopwords = FALSE, ...)
  as.character(toks[[1]])
}

test_that(".normalize_math_notation folds LaTeX markup to plain grammar", {
  n <- TextAnalysisR:::.normalize_math_notation
  expect_equal(n("\\frac{3}{4}"), "3/4")
  expect_equal(n("\\sqrt{16}"), "sqrt(16)")
  expect_equal(n("x^{2}"), "x^2")
  expect_equal(n("y^(3)"), "y^3")
  expect_equal(n("a \\times b"), "a * b")
  expect_equal(n("a \\cdot b"), "a * b")
})

test_that(".normalize_math_notation folds LaTeX and unicode operators alike", {
  n <- TextAnalysisR:::.normalize_math_notation
  expect_equal(n("x \\leq 5"), "x <= 5")
  expect_equal(n("x ≤ 5"), "x <= 5")
  expect_equal(n("x \\geq 5"), "x >= 5")
  expect_equal(n("x ≥ 5"), "x >= 5")
  expect_equal(n("x \\neq 5"), "x != 5")
  expect_equal(n("x ≠ 5"), "x != 5")
  expect_equal(n("3 × 4"), "3 * 4")
  expect_equal(n("8 ÷ 2"), "8 / 2")
  expect_equal(n("5 \\pm 1"), "5 +/- 1")
})

test_that(".normalize_math_notation handles empty, NA, and plain text", {
  n <- TextAnalysisR:::.normalize_math_notation
  expect_equal(n(character(0)), character(0))
  expect_true(is.na(n(NA_character_)))
  expect_equal(n("plain sentence"), "plain sentence")
})

test_that("math mode keeps multi-character operators as single tokens", {
  skip_if_not_installed("quanteda")
  toks <- .mm_tokens("if y >= 2 and y <= 10 then stop", math_mode = TRUE)
  expect_true(">=" %in% toks)
  expect_true("<=" %in% toks)
  expect_false("=" %in% toks)
})

test_that("math mode preserves grouping so different expressions differ", {
  skip_if_not_installed("quanteda")
  a <- .mm_tokens("(x + 1) * 2", math_mode = TRUE)
  b <- .mm_tokens("x + (1 * 2)", math_mode = TRUE)
  expect_true("(" %in% a)
  expect_false(identical(a, b))
})

test_that("math mode strips sentence punctuation but keeps operators", {
  skip_if_not_installed("quanteda")
  toks <- .mm_tokens("Solve x + 3 = 4, then stop.", math_mode = TRUE)
  expect_false("." %in% toks)
  expect_false("," %in% toks)
  expect_true("+" %in% toks)
  expect_true("=" %in% toks)
})

test_that("math mode keeps numbers and single-character tokens", {
  skip_if_not_installed("quanteda")
  toks <- .mm_tokens("Angle A measures 45 degrees and b = 5 m", math_mode = TRUE)
  expect_true("45" %in% toks)
  expect_true("a" %in% toks)
  expect_true("m" %in% toks)
})

test_that("math mode normalizes LaTeX before tokenizing", {
  skip_if_not_installed("quanteda")
  toks <- .mm_tokens("Compute \\frac{3}{4} and x \\leq 5", math_mode = TRUE)
  expect_false("\\" %in% toks)
  expect_false("frac" %in% toks)
  expect_true("<=" %in% toks)
})

test_that("default mode is unchanged by the math-mode additions", {
  skip_if_not_installed("quanteda")
  toks <- .mm_tokens("Solve x + 3 = 4 now", math_mode = FALSE)
  expect_false("+" %in% toks)
  expect_false("=" %in% toks)
  expect_true("solve" %in% toks)
})
