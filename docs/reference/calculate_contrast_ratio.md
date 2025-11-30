# Web Accessibility Utility Functions

Functions for ensuring WCAG 2.1 Level AA compliance in the Shiny
application

Calculates the contrast ratio between two colors according to WCAG 2.1
standards using the relative luminance formula from W3C guidelines. Used
to verify text/background color combinations meet accessibility
requirements.

## Usage

``` r
calculate_contrast_ratio(foreground, background)
```

## Arguments

- foreground:

  Foreground color (hex format, e.g., "#111827")

- background:

  Background color (hex format, e.g., "#ffffff")

## Value

Numeric contrast ratio (1-21)

## WCAG 2.1 Level AA Compliance

This package follows Web Content Accessibility Guidelines (WCAG) 2.1
Level AA:

- 1.1.1 Non-text Content (Level A): Alt text for images and
  visualizations

- 1.4.3 Contrast Minimum (Level AA): 4.5:1 ratio for normal text, 3:1
  for large text/UI

- 2.1.1 Keyboard (Level A): Full keyboard navigation support

- 2.4.1 Bypass Blocks (Level A): Skip navigation links

- 3.1.1 Language of Page (Level A): Page language identification

- 4.1.2 Name, Role, Value (Level A): ARIA labels and roles Calculate
  Color Contrast Ratio

## WCAG Requirements

- Normal text: Minimum 4.5:1 (Level AA)

- Large text (18pt+ or 14pt+ bold): Minimum 3:1 (Level AA)

- UI components and graphics: Minimum 3:1 (Level AA)

## Examples

``` r
if (FALSE) { # \dontrun{
calculate_contrast_ratio("#111827", "#ffffff")  # Returns ~16:1 (Pass)
calculate_contrast_ratio("#6b7280", "#4a5568")  # Returns ~2.8:1 (Fail)
} # }
```
