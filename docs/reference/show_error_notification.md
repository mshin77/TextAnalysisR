# Show error notification

Show error notification

## Usage

``` r
show_error_notification(message, duration = 7, session = NULL)
```

## Arguments

- message:

  Notification text.

- duration:

  Seconds until auto-dismiss.

- session:

  Shiny session. Defaults to the current reactive domain.

## Value

Invisibly the notification id.
