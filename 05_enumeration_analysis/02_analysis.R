# 02_analysis_enumeration_vs_soluble.R
# Performs univariate summaries, statistical tests, and effect-size calculations
# on a merged 'all_libs' CSV

# 1. Load required packages
library(tidyverse)
library(broom)


# Load new combined data with 8 libraries
input_path <- "C:/Users/Admin/Documents/Hu Lab/Code/Enumeration Library Code/Library_Analysis_Code/all_libs_Enumeration_Linker_Diversity.csv"
all_libs <- read_csv(input_path, show_col_types = FALSE)
all_libs <- all_libs %>%
  mutate(Library = factor(Library, levels = c(
    "Meta",
    "trans-1,4-dibromo-2-butene",
    "2,3-bis-bromomethyl-quinoxaline",
    "2,6-bis-bromomethyl-naphthalene",
    "3,4-bis-bromomethyl-furan",
    "4,6-divinylpyrimidin-2-amine",
    "dibromo-pyridine",
    "divinyl-sulfone"
  )))

# 3. Univariate summary statistics ------------------------------
summary_tbl <- all_libs %>%
  pivot_longer(
    cols      = c(Total.Molweight, cLogP, cLogS, H.Acceptors, H.Donors, Rotatable.Bonds, Polar.Surface.Area, aLogP),   # add your RDKit column here if present
    names_to  = "Descriptor",
    values_to = "Value"
  ) %>%
  group_by(Library, Descriptor) %>%
  summarise(
    N       = n(),
    Missing = sum(is.na(Value)),
    Mean    = mean(Value,   na.rm = TRUE),
    Median  = median(Value, na.rm = TRUE),
    SD      = sd(Value,     na.rm = TRUE),
    .groups = "drop"
  )

out_prefix <- "Linker_Diversity"

write_csv(summary_tbl, paste0("summary_", out_prefix, ".csv"))
print(summary_tbl)

# 4. Statistical tests ------------------------------------------
library(purrr)

# Pairwise statistical tests for all library combinations
desc_names <- c("Total.Molweight", "cLogP", "cLogS", "H.Acceptors", "H.Donors", "Rotatable.Bonds", "Polar.Surface.Area", "aLogP")
lib_names <- levels(all_libs$Library)
pairwise_tests <- function(df, desc, lib1, lib2) {
  x <- df$Value[df$Library == lib1 & !is.na(df$Value)]
  y <- df$Value[df$Library == lib2 & !is.na(df$Value)]
  tibble(
    Descriptor = desc,
    Library1 = lib1,
    Library2 = lib2,
    t_p = if(length(x) > 1 && length(y) > 1) t.test(x, y)$p.value else NA_real_,
    wilcox_p = if(length(x) > 0 && length(y) > 0) wilcox.test(x, y)$p.value else NA_real_,
    ks_p = if(length(x) > 0 && length(y) > 0) ks.test(x, y, exact = FALSE)$p.value else NA_real_
  )
}

test_tbl <- map_dfr(desc_names, function(desc) {
  df <- all_libs %>% select(Library, !!sym(desc)) %>% rename(Value = !!sym(desc))
  combn(lib_names, 2, function(libs) pairwise_tests(df, desc, libs[1], libs[2]), simplify = FALSE) %>% bind_rows()
})

write_csv(test_tbl, paste0("tests_", out_prefix, ".csv"))
print(test_tbl, digits = 30)

# 5. Effect sizes (Cohen's d) -----------------------------------

# Pairwise Cohen's d for all library combinations
pairwise_cohen_d <- function(df, desc, lib1, lib2) {
  x <- df$Value[df$Library == lib1 & !is.na(df$Value)]
  y <- df$Value[df$Library == lib2 & !is.na(df$Value)]
  mean_x <- mean(x)
  mean_y <- mean(y)
  sd_x <- sd(x)
  sd_y <- sd(y)
  n_x <- length(x)
  n_y <- length(y)
  pooled_sd <- sqrt(((n_x - 1) * sd_x^2 + (n_y - 1) * sd_y^2) / (n_x + n_y - 2))
  cohen_d <- (mean_x - mean_y) / pooled_sd
  tibble(
    Descriptor = desc,
    Library1 = lib1,
    Library2 = lib2,
    Mean1 = mean_x,
    Mean2 = mean_y,
    pooled_sd = pooled_sd,
    cohen_d = cohen_d
  )
}

effect_tbl <- map_dfr(desc_names, function(desc) {
  df <- all_libs %>% select(Library, !!sym(desc)) %>% rename(Value = !!sym(desc))
  combn(lib_names, 2, function(libs) pairwise_cohen_d(df, desc, libs[1], libs[2]), simplify = FALSE) %>% bind_rows()
})

write_csv(effect_tbl, paste0("effect_sizes_", out_prefix, ".csv"))
print(effect_tbl)
