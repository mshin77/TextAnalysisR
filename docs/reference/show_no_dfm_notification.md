# Show No DFM Notification

Displays a standardized error notification when DFM is required but not
available. Shorter alternative to the modal dialog for simple error
messages.

## Usage

``` r
show_no_dfm_notification(feature_name = "this feature", duration = 7)
```

## Arguments

- feature_name:

  Name of the feature requiring DFM (default: "this feature")

- duration:

  Duration in seconds (default: 7)

## Value

Displays a Shiny notification. Returns NULL invisibly.
