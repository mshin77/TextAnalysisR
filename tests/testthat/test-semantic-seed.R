test_that("reduce_dimensions does not leak .Random.seed to caller", {
  skip_if_not_installed("withr")
  withr::with_seed(42, {
    m <- matrix(rnorm(40), nrow = 10, ncol = 4)
    pre_seed <- .Random.seed
    suppressMessages(
      reduce_dimensions(
        m,
        method = "PCA",
        n_components = 2,
        seed = 7,
        verbose = FALSE
      )
    )
    expect_identical(.Random.seed, pre_seed)
  })
})

test_that("reduce_dimensions is reproducible across seeded calls", {
  skip_if_not_installed("withr")
  m <- matrix(rnorm(40), nrow = 10, ncol = 4)
  a <- suppressMessages(
    reduce_dimensions(m, method = "PCA", n_components = 2, seed = 11, verbose = FALSE)
  )
  b <- suppressMessages(
    reduce_dimensions(m, method = "PCA", n_components = 2, seed = 11, verbose = FALSE)
  )
  expect_equal(a$reduced_data, b$reduced_data)
})
