# Show Loading/Progress Notification

Displays a persistent loading notification with a specific ID that can
be removed later.

## Usage

``` r
show_loading_notification(message, id = NULL)
```

## Arguments

- message:

  The loading message to display

- id:

  Notification ID for later removal (optional)

## Value

Displays a Shiny notification. Returns NULL invisibly.
