# Show Feature Matrix Notification

Displays error notification when feature matrix is required but not
available. Similar to show_no_dfm_notification but uses "feature matrix"
terminology.

## Usage

``` r
show_no_feature_matrix_notification(duration = 7)
```

## Arguments

- duration:

  Duration in seconds (default: 7)

## Value

Displays a Shiny notification. Returns NULL invisibly.
