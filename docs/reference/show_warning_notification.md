# Show warning notification

Show warning notification

## Usage

``` r
show_warning_notification(message, duration = 5, session = NULL)
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
