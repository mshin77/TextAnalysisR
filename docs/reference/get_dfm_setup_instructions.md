# Generate DFM Setup Instructions Text

Generates standardized text instructions for creating a DFM. Used in
console output or verbatim text displays.

## Usage

``` r
get_dfm_setup_instructions(feature_name = "this feature")
```

## Arguments

- feature_name:

  Name of the feature requiring DFM (default: "this feature")

## Value

Character vector of instruction lines

## Examples

``` r
instructions <- get_dfm_setup_instructions("keyword extraction")
cat(instructions, sep = "\n")
#> Warning: DFM Processing Required
#> 
#> Please complete the following steps first:
#> 
#> 1. Go to the 'Preprocess' tab
#> 2. Navigate to Step 4: Document-Feature Matrix
#> 3. Click the 'Process' button
#> 
#> Once the DFM is created, you can return here to use keyword extraction.
```
