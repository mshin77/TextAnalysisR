# Complete Text Mining Workflow

This function provides a complete text mining workflow that follows the
same sequence as the Shiny application: file processing -\> text uniting
-\> preprocessing -\> DFM creation -\> analysis. It serves as a
convenience function for users who want to execute the entire pipeline
programmatically.

## Usage

``` r
run_text_workflow(
  dataset_choice,
  file_info = NULL,
  text_input = NULL,
  listed_vars,
  min_char = 2,
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_numbers = TRUE,
  remove_url = TRUE,
  detect_compounds = FALSE,
  compound_size = 2:3,
  compound_min_count = 2,
  verbose = TRUE
)
```

## Arguments

- dataset_choice:

  A character string indicating the dataset choice: "Upload an Example
  Dataset", "Upload Your File", "Copy and Paste Text".

- file_info:

  A data frame containing file information (for file upload).

- text_input:

  A character string containing text input (for copy-paste).

- listed_vars:

  A character vector of column names to unite into text.

- min_char:

  The minimum number of characters for tokens (default: 2).

- remove_punct:

  Logical; remove punctuation (default: TRUE).

- remove_symbols:

  Logical; remove symbols (default: TRUE).

- remove_numbers:

  Logical; remove numbers (default: TRUE).

- remove_url:

  Logical; remove URLs (default: TRUE).

- detect_compounds:

  Logical; detect multi-word expressions (default: FALSE).

- compound_size:

  Size range for compound detection (default: 2:3).

- compound_min_count:

  Minimum count for compounds (default: 2).

- verbose:

  Logical; print progress messages (default: TRUE).

## Value

A list containing processed data, tokens, DFM, and metadata.

## Examples

``` r
# \donttest{
  workflow_result <- TextAnalysisR::run_text_workflow(
    dataset_choice = "Upload an Example Dataset",
    listed_vars = c("title", "keyword", "abstract")
  )
#> Starting complete text mining workflow...
#> Step 1: Processing files...
#> Step 2: Uniting text columns...
#> Step 3: Preprocessing texts...
#> Creating corpus...
#> Tokenizing texts...
#> Creating a tokens from a corpus object...
#>  ...starting tokenization
#>  ...tokenizing 1 of 1 blocks
#>  ...preserving elisions
#>  ...removing separators, punctuation, symbols, numbers, URLs 
#>  ...6,736 unique types
#>  ...complete, elapsed time: 0.31 seconds.
#> Finished constructing tokens from 490 documents
#> Converting to lowercase...
#> Applying minimum character filter...
#> Text preprocessing completed in 0.45 seconds
#> Processed 490 documents with 1218465601403578929318175265121813953461568396368196169611121638337616515867194144150377958690910316623182406173226381248722423928817422019313711774198121165879777161148193166981339668127659915512511214711817592104375815923753691181572861919514618299657911375921213617316320816416210841886271146133752381481664241591324416346172146136172151682572583011491701161751513884951341561262932296889300131156181161157244142277316533159190251213146165152133113262397274138248190285181763261851791801043897415118411431544841813037133610615217719311818111927433117917735721018010928118017713219516432012219026714518016713812118315512326429213247014811915717320424123114520335832033317813928013918226227527841293247164191187166204180257246158168137227222216316194235203188192262197173148261263187201108186135375231154831851472195814618327118516317324628618925816535517316718117524835830042127622021316917732124714914130219215618316115112518317940716421126822219316421114518112927425316233636842919817126730826916535215521717220419027814218118840727712210623921320618618811217218520021518515017418418315619016521517017212315424116714153627326820016917618729219818032628319732812213013615312413414418318415319421392249216387199232158311170205195202230143187238485215396258156147206266143244318134194164215201116138150184275170264291197222147126188201213246142144156193141246159138270226122184172156151181152162153126189236193136 total tokens
#> Step 5: Creating document-feature matrix...
#> Complete text mining workflow finished in 1.17 seconds
#> Documents processed: 490
#> Features identified: 5078
# }
if (interactive()) {
  file_info <- data.frame(filepath = "path/to/file.xlsx")
  workflow_result <- TextAnalysisR::run_text_workflow(
    dataset_choice = "Upload Your File",
    file_info = file_info,
    listed_vars = c("column1", "column2")
  )

  workflow_result <- TextAnalysisR::run_text_workflow(
    dataset_choice = "Copy and Paste Text",
    text_input = "Your text content here",
    listed_vars = "text"
  )
}
```
