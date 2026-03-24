library(tidyverse)

# 1. Read each effect‐size table and tag it
orig_pol   <- read_csv("effect_sizes_polar_vs_original.csv")       %>% mutate(Comparison="Original_vs_Polar")
pol_ser    <- read_csv("effect_sizes_polar_vs_ser.csv")            %>% mutate(Comparison="Polar_vs_Ser")
ser_sol    <- read_csv("effect_sizes_ser_vs_soluble.csv")          %>% mutate(Comparison="Ser_vs_Soluble")
mono_sol   <- read_csv("effect_sizes_monomer_vs_soluble.csv")      %>% mutate(Comparison="Monomer_vs_Soluble")
orig_sol   <- read_csv("effect_sizes_Dipeptide_vs_soluble.csv")     %>% mutate(Comparison="Original_vs_Soluble")
mono_orig  <- read_csv("effect_sizes_monomer_vs_original.csv")     %>% mutate(Comparison="Monomer_vs_Original")

# 2. Combine them all and flip the sign only for Monomer_vs_Original
all_eff <- bind_rows(
  orig_pol, pol_ser, ser_sol,
  mono_sol, orig_sol, mono_orig
) %>%
  mutate(
    cohen_d = if_else(
      Comparison == "Monomer_vs_Original",
      -cohen_d,
      cohen_d
    )
  )

# 3. Reorder factor levels
all_eff <- all_eff %>%
  mutate(
    Descriptor = factor(Descriptor, levels = c(
      "H.Acceptors","H.Donors","Polar.Surface.Area",
      "Rotatable.Bonds","Total.Molweight","cLogP","cLogS"
    )),
    Comparison = factor(Comparison, levels = c(
      "Original_vs_Polar",
      "Polar_vs_Ser",
      "Ser_vs_Soluble",
      "Monomer_vs_Soluble",
      "Original_vs_Soluble",
      "Monomer_vs_Original"
    ))
  )

# 4. Build the bar‐chart
p <- ggplot(all_eff, aes(x = Descriptor, y = cohen_d, fill = Comparison)) +
  geom_col(position = "dodge") +
  coord_flip() +
  theme_minimal(base_size = 14) +
  scale_fill_brewer(palette = "Dark2") +
  labs(
    title = "Cohen’s d Across Library Comparisons",
    x     = NULL,
    y     = "Cohen’s d"
  )

# 5. Display
print(p)

# 6. Save
ggsave("effect_sizes_comparison_6way.png", p, width = 9, height = 6, units = "in", dpi = 300)
ggsave("effect_sizes_comparison_6way.pdf", p, width = 9, height = 6, units = "in", device = cairo_pdf)

