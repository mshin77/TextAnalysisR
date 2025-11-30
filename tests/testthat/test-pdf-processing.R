test_that("process_pdf_unified validates file existence", {
  result <- process_pdf_unified("nonexistent.pdf")

  expect_false(result$success)
  expect_equal(result$type, "error")
  expect_equal(result$method, "none")
  expect_match(result$message, "File not found")
})

test_that("process_pdf_unified falls back to R when Python unavailable", {
  skip_if_not_installed("pdftools")
  skip_if(!file.exists("test-sample.pdf"))

  result <- process_pdf_unified(
    "test-sample.pdf",
    use_multimodal = FALSE
  )

  expect_type(result, "list")
  expect_true("success" %in% names(result))
  expect_true("method" %in% names(result))
  expect_true("data" %in% names(result))
})

test_that("import_files handles DOCX text extraction", {
  skip_if_not_installed("officer")
  skip_if(!file.exists("test-sample.docx"))

  file_info <- data.frame(
    filepath = "test-sample.docx",
    stringsAsFactors = FALSE
  )

  result <- import_files(
    dataset_choice = "Upload Your File",
    file_info = file_info
  )

  expect_s3_class(result, "tbl_df")
  expect_true("text" %in% names(result))
  expect_true(nrow(result) > 0)
})

test_that("import_files handles CSV files", {
  temp_csv <- tempfile(fileext = ".csv")
  write.csv(
    data.frame(text = c("Test1", "Test2"), stringsAsFactors = FALSE),
    temp_csv,
    row.names = FALSE
  )

  file_info <- data.frame(filepath = temp_csv, stringsAsFactors = FALSE)
  result <- import_files("Upload Your File", file_info = file_info)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 2)

  unlink(temp_csv)
})

test_that("import_files handles TXT files", {
  temp_txt <- tempfile(fileext = ".txt")
  writeLines(c("Line 1", "Line 2"), temp_txt)

  file_info <- data.frame(filepath = temp_txt, stringsAsFactors = FALSE)
  result <- import_files("Upload Your File", file_info = file_info)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 2)

  unlink(temp_txt)
})

test_that("import_files validates empty files", {
  temp_txt <- tempfile(fileext = ".txt")
  writeLines("", temp_txt)

  file_info <- data.frame(filepath = temp_txt, stringsAsFactors = FALSE)
  result <- import_files("Upload Your File", file_info = file_info)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0)

  unlink(temp_txt)
})
