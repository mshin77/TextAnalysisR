# Process Files

This function processes different types of files and text input based on
the dataset choice.

## Usage

``` r
import_files(dataset_choice, file_info = NULL, text_input = NULL)
```

## Arguments

- dataset_choice:

  A character string indicating the dataset choice.

- file_info:

  A data frame containing file information with a column named
  'filepath' (default: NULL).

- text_input:

  A character string containing text input (default: NULL).

## Value

A data frame containing the processed data.

## Examples

``` r
if (interactive()) {
  mydata <- TextAnalysisR::SpecialEduTech
  mydata <- TextAnalysisR::import_files(dataset_choice = "Upload an Example Dataset")
  head(mydata)

  file_info <- data.frame(filepath = "inst/extdata/SpecialEduTech.xlsx")
  mydata <- TextAnalysisR::import_files(dataset_choice = "Upload Your File",
                                          file_info = file_info)
  head(mydata)


  text_input <- paste("Virtual manipulatives for algebra instruction",
                      "manipulatives mathematics learning disability",
                      "This study examined virtual manipulatives effects on",
                      "students with learning disabilities")
  mydata <- TextAnalysisR::import_files(dataset_choice = "Copy and Paste Text",
                                          text_input = text_input)
  head(mydata)
}
```
