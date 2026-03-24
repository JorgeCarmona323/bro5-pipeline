library(tidyverse)

# 1. Read the effect size table and tag it
mono_dipep <- read_csv("effect_sizes_Monomer_vs_Dipeptide.csv") %>% mutate(Comparison="Monomer_vs_Dipeptide")

# 2. Reorder factor levels
all_eff <- mono_dipep %>%
  mutate(
    Descriptor = factor(Descriptor, levels = c(
      "H.Acceptors","H.Donors","Polar.Surface.Area",
      "Rotatable.Bonds","Total.Molweight","cLogP","cLogS"
    )),
    Comparison = factor(Comparison, levels = c(
      "Monomer_vs_Dipeptide"
    ))
  )

# 3. Build the bar chart
p <- ggplot(all_eff, aes(x = Descriptor, y = cohen_d, fill = Comparison)) +
  geom_col(position = "dodge") +
  coord_flip() +
  theme_minimal(base_size = 14) +
  scale_fill_brewer(palette = "Dark2") +
  labs(
    title = "Cohen's d: Monomer vs Dipeptide",
    x     = NULL,
    y     = "Cohen's d"
  )

# 4. Display
print(p)

# 5. Save in SVG format only
ggsave("effect_sizes_monomer_vs_dipeptide.svg", p, width = 9, height = 6, units = "in")

# 1. Define your descriptor columns
desc_cols <- c("Total.Molweight","cLogP","cLogS",
               "H.Acceptors","H.Donors","Rotatable.Bonds","Polar.Surface.Area")

# 2. Compute Original centroid & pooled covariance
orig_cent  <- colMeans(filter(all_libs, Library=="Dipeptide")[desc_cols])
cov_pooled <- cov(all_libs[desc_cols])

# 3. Compute Mahalanobis distance for each compound
all_libs_dist <- all_libs %>%
  rowwise() %>%
  mutate(
    Mdist = mahalanobis(
      x      = c_across(all_of(desc_cols)),
      center = orig_cent,
      cov    = cov_pooled
    ),
    sqrtM  = sqrt(Mdist)
  ) %>%
  ungroup()

# 4. Build the density plot
p_md <- ggplot(all_libs_dist, aes(x = sqrtM, fill = Library)) +
  geom_density(alpha = 0.3) +
  theme_minimal(base_size = 14) +
  scale_fill_brewer(palette = "Dark2") +
  labs(
    title = "Distance from Dipeptide Centroid (Mahalanobis)",
    x     = "√Mahalanobis Distance",
    y     = "Density"
  )

# 5. Display
print(p_md)

# 6. Save as SVG
ggsave("mahalanobis_distance_density.svg", p_md, width=8, height=6, units="in")
