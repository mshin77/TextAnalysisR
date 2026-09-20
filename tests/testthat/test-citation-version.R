cited_version <- function(path) {
  txt <- readLines(path, warn = FALSE)
  hit <- grep("package version [0-9]", txt, value = TRUE)
  sub(".*package version ([0-9][^)]*)\\).*", "\\1", hit)
}

test_that("the citation names one version everywhere", {
  readme <- system.file("..", "README.md", package = "TextAnalysisR")
  about <- system.file("TextAnalysisR.app/markdown/about.md", package = "TextAnalysisR")
  skip_if_not(file.exists(readme) && nzchar(about) && file.exists(about),
              "citation sources not installed")

  in_readme <- cited_version(readme)
  in_about <- cited_version(about)

  expect_length(in_readme, 1)
  expect_length(in_about, 1)
  expect_identical(in_about, in_readme)
})
