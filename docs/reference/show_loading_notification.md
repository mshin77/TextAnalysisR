# Show persistent loading notification

Show persistent loading notification

## Usage

``` r
show_loading_notification(message, id = NULL, session = NULL)
```

## Arguments

- message:

  Notification text.

- id:

  Optional unique id for later removal.

- session:

  Shiny session. Defaults to the current reactive domain.

## Value

Invisibly the notification id.
