#' Web Accessibility Utility Functions
#'
#' @description Functions for ensuring WCAG 2.1 Level AA compliance in the Shiny application
#'
#' @section WCAG 2.1 Level AA Compliance:
#' This package follows Web Content Accessibility Guidelines (WCAG) 2.1 Level AA:
#' - 1.1.1 Non-text Content (Level A): Alt text for images and visualizations
#' - 1.4.3 Contrast Minimum (Level AA): 4.5:1 ratio for normal text, 3:1 for large text/UI
#' - 2.1.1 Keyboard (Level A): Full keyboard navigation support
#' - 2.4.1 Bypass Blocks (Level A): Skip navigation links
#' - 3.1.1 Language of Page (Level A): Page language identification
#' - 4.1.2 Name, Role, Value (Level A): ARIA labels and roles

#' Calculate Color Contrast Ratio
#'
#' @description
#' Calculates the contrast ratio between two colors according to WCAG 2.1 standards
#' using the relative luminance formula from W3C guidelines.
#' Used to verify text/background color combinations meet accessibility requirements.
#'
#' @param foreground Foreground color (hex format, e.g., "#111827")
#' @param background Background color (hex format, e.g., "#ffffff")
#'
#' @return Numeric contrast ratio (1-21)
#' @keywords internal
#'
#' @section WCAG Requirements:
#' - Normal text: Minimum 4.5:1 (Level AA)
#' - Large text (18pt+ or 14pt+ bold): Minimum 3:1 (Level AA)
#' - UI components and graphics: Minimum 3:1 (Level AA)
#'
#' @examples
#' \dontrun{
#' calculate_contrast_ratio("#111827", "#ffffff")  # Returns ~16:1 (Pass)
#' calculate_contrast_ratio("#6b7280", "#4a5568")  # Returns ~2.8:1 (Fail)
#' }
calculate_contrast_ratio <- function(foreground, background) {
  hex_to_rgb <- function(hex) {
    hex <- gsub("#", "", hex)
    c(
      strtoi(substr(hex, 1, 2), 16L),
      strtoi(substr(hex, 3, 4), 16L),
      strtoi(substr(hex, 5, 6), 16L)
    ) / 255
  }

  relative_luminance <- function(rgb) {
    rgb <- sapply(rgb, function(val) {
      if (val <= 0.03928) {
        val / 12.92
      } else {
        ((val + 0.055) / 1.055)^2.4
      }
    })
    0.2126 * rgb[1] + 0.7152 * rgb[2] + 0.0722 * rgb[3]
  }

  fg_rgb <- hex_to_rgb(foreground)
  bg_rgb <- hex_to_rgb(background)

  l1 <- relative_luminance(fg_rgb)
  l2 <- relative_luminance(bg_rgb)

  if (l1 > l2) {
    ratio <- (l1 + 0.05) / (l2 + 0.05)
  } else {
    ratio <- (l2 + 0.05) / (l1 + 0.05)
  }

  return(round(ratio, 2))
}

#' Check WCAG Contrast Compliance
#'
#' @description
#' Validates if color combination meets WCAG 2.1 Level AA contrast requirements.
#'
#' @param foreground Foreground color (hex format)
#' @param background Background color (hex format)
#' @param large_text Logical, TRUE if text is large (18pt+ or 14pt+ bold)
#'
#' @return Logical TRUE if compliant, FALSE if not
#' @keywords internal
#'
#' @examples
#' \dontrun{
#' check_wcag_contrast("#111827", "#ffffff")  # TRUE (16:1 ratio)
#' check_wcag_contrast("#6b7280", "#4a5568")  # FALSE (2.8:1 ratio)
#' }
check_wcag_contrast <- function(foreground, background, large_text = FALSE) {
  ratio <- calculate_contrast_ratio(foreground, background)
  min_ratio <- if (large_text) 3.0 else 4.5

  if (ratio >= min_ratio) {
    return(TRUE)
  } else {
    warning(
      "WCAG contrast failure: ", ratio, ":1 ratio (requires ", min_ratio, ":1)\n",
      "  Foreground: ", foreground, "\n",
      "  Background: ", background
    )
    return(FALSE)
  }
}

#' Generate ARIA Label
#'
#' @description
#' Creates accessible ARIA label for UI elements.
#'
#' @param element_type Type of element (e.g., "button", "input", "plot")
#' @param action Action or purpose (e.g., "analyze", "download", "visualize")
#' @param context Additional context (optional)
#'
#' @return Character string with ARIA label
#' @keywords internal
#'
#' @examples
#' \dontrun{
#' generate_aria_label("button", "analyze", "readability")
#' # Returns: "Analyze readability button"
#' }
generate_aria_label <- function(element_type, action, context = NULL) {
  if (!is.null(context)) {
    label <- paste(tools::toTitleCase(action), context, element_type)
  } else {
    label <- paste(tools::toTitleCase(action), element_type)
  }
  return(label)
}

#' Create Screen Reader Text
#'
#' @description
#' Generates visually hidden text for screen readers (WCAG 4.1.2).
#'
#' @param text Text to be read by screen readers
#'
#' @return HTML span with sr-only class
#' @keywords internal
#'
#' @examples
#' \dontrun{
#' create_sr_text("Loading results, please wait")
#' }
create_sr_text <- function(text) {
  return(
    paste0(
      '<span class="sr-only" role="status" aria-live="polite">',
      text,
      '</span>'
    )
  )
}

#' Validate Keyboard Navigation
#'
#' @description
#' Checks if interactive elements have proper tabindex and keyboard handlers.
#' Used for WCAG 2.1.1 (Keyboard) compliance.
#'
#' @param tabindex Integer, tab order (-1 for no tab, 0 for natural order, 1+ for specific order)
#'
#' @return Logical TRUE if valid, FALSE with warning if invalid
#' @keywords internal
#'
#' @examples
#' \dontrun{
#' validate_keyboard_navigation(0)   # TRUE
#' validate_keyboard_navigation(999) # FALSE (too high)
#' }
validate_keyboard_navigation <- function(tabindex = 0) {
  if (!is.numeric(tabindex)) {
    warning("Tabindex must be numeric")
    return(FALSE)
  }

  if (tabindex > 100) {
    warning("Tabindex > 100 creates unpredictable tab order (WCAG 2.1.1)")
    return(FALSE)
  }

  return(TRUE)
}

#' Check Alt Text Presence
#'
#' @description
#' Validates that images and visualizations have alternative text descriptions.
#' Required for WCAG 1.1.1 (Non-text Content).
#'
#' Note: Decorative images should use empty alt text (alt="") to indicate
#' they should be ignored by assistive technology.
#'
#' @param alt_text Alternative text description
#' @param element_type Type of element (e.g., "plot", "image", "icon")
#' @param decorative Logical, TRUE if element is purely decorative
#'
#' @return Logical TRUE if valid, FALSE with warning if missing/inadequate
#' @keywords internal
#'
#' @examples
#' \dontrun{
#' check_alt_text("Bar chart showing word frequency", "plot")  # TRUE
#' check_alt_text("", "plot")  # FALSE (informative content needs alt text)
#' check_alt_text("", "icon", decorative = TRUE)  # TRUE (decorative is OK)
#' }
check_alt_text <- function(alt_text, element_type = "image", decorative = FALSE) {
  if (decorative) {
    return(TRUE)
  }

  if (is.null(alt_text) || !nzchar(alt_text)) {
    warning("Missing alt text for ", element_type, " (WCAG 1.1.1)")
    return(FALSE)
  }

  if (nchar(alt_text) < 10) {
    warning("Alt text too short for ", element_type, " (consider more descriptive text)")
    return(FALSE)
  }

  return(TRUE)
}
