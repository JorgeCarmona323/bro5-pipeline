# 03_plots_monomer_vs_dipeptide.R
library(tidyverse)
library(ggthemes)
library(patchwork)
library(cowplot)
library(RColorBrewer)

all_libs <- read_csv("c:/Users/Admin/Documents/Hu Lab/DataWarrior/Linker_Enumeration_09152025/all_libs_Linker_Diversity_aLogP.csv", show_col_types = FALSE)
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

# descriptors now include aLogP
desc_cols <- c("Total.Molweight", "cLogP", "cLogS",
               "H.Acceptors", "H.Donors", "Rotatable.Bonds",
               "Polar.Surface.Area", "aLogP")

descriptors <- list(
  Total.Molweight    = list(var="Total.Molweight",    bin=40,  xlab="Mol. Weight (Da)", xlim=c(600,1600), breaks=c(800,1200,1600)),
  cLogP              = list(var="cLogP",              bin=0.7, xlab="cLogP"),
  cLogS              = list(var="cLogS",              bin=0.5, xlab="cLogS"),
  H.Acceptors        = list(var="H.Acceptors",        bin=1,   xlab="H-bond Acceptors"),
  H.Donors           = list(var="H.Donors",           bin=1,   xlab="H-bond Donors"),
  Rotatable.Bonds    = list(var="Rotatable.Bonds",    bin=1,   xlab="Rotatable Bonds"),
  Polar.Surface.Area = list(var="Polar.Surface.Area", bin=20,  xlab="TPSA (Å²)"),
  aLogP              = list(var="aLogP",              bin=0.5, xlab="aLogP (RDKit)")
)

# Histogram helper with orange/cyan
lib_colors <- c(
  Meta = "#56B4E9",                                 # blue
  "trans-1,4-dibromo-2-butene" = "#E69F00",         # orange
  "2,3-bis-bromomethyl-quinoxaline" = "#009E73",    # green
  "2,6-bis-bromomethyl-naphthalene" = "#F0E442",    # yellow
  "3,4-bis-bromomethyl-furan" = "#D55E00",          # reddish
  "4,6-divinylpyrimidin-2-amine" = "#CC79A7",       # purple
  "dibromo-pyridine" = "#0072B2",                   # dark blue
  "divinyl-sulfone" = "#999999"                     # grey
)

make_histogram <- function(df, settings) {
  p <- ggplot(df, aes(x = .data[[settings$var]], fill = Library)) +
    geom_histogram(alpha = 0.6, position = "identity",
                   binwidth = settings$bin, color = "black", linewidth = 0.3, na.rm=TRUE) +
    theme_tufte(base_family = "sans", base_size = 12) +
    labs(x=settings$xlab, y="Count") +
    scale_fill_manual(values = lib_colors) +
    theme(legend.position = "none",
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10))
  if (!is.null(settings$xlim)) {
    if (!is.null(settings$breaks)) {
      p <- p + scale_x_continuous(limits=settings$xlim, breaks=settings$breaks)
    } else {
      p <- p + scale_x_continuous(limits=settings$xlim)
    }
  }
  p
}

# Build histograms
histogram_plots <- imap(descriptors, ~ make_histogram(all_libs, .x))

# Legend with orange/cyan
legend_plot <- ggplot(all_libs, aes(x = Total.Molweight, fill = Library)) +
  geom_histogram(alpha = 0.6, binwidth=40) +
  scale_fill_manual(values = lib_colors) +
  theme_void() +
  theme(legend.position = "right",
    legend.title = element_text(size = 10, family = "sans"),
    legend.text  = element_text(size = 9, family = "sans"),
    legend.key.size = unit(0.5, "cm")) +
  guides(fill = guide_legend(title = "Library"))

combined <- wrap_plots(
  wrap_plots(histogram_plots, ncol = 4),
  get_legend(legend_plot),
  ncol = 2,
  widths = c(8, 1)
)
print(combined)
ggsave("Linker_Diversity_histograms.svg", combined, width=14, height=8, units="in")

# Rainclouds
make_raincloud <- function(df, settings) {
  ggplot(df, aes(x=Library, y=.data[[settings$var]], fill=Library)) +
    geom_violin(trim=FALSE, alpha=0.4, na.rm=TRUE) +
    geom_boxplot(width=0.15, outlier.size=0.7, na.rm=TRUE, color="black") +
    theme_bw(base_family = "sans", base_size = 12) +
    labs(x=NULL, y=settings$xlab) +
    scale_fill_manual(values = lib_colors) +
    theme(legend.position = "right",
          legend.title = element_text(size = 10, family = "sans"),
          legend.text  = element_text(size = 9, family = "sans"))
}
rainclouds <- imap(descriptors, ~ make_raincloud(all_libs, .x))
combined_rain <- wrap_plots(rainclouds, ncol=2) +
  plot_annotation(title="Monomer vs. Dipeptide: Rainclouds") &
  theme(legend.position="right")
print(combined_rain)
ggsave("Linker_Diversity_raincloud.svg", combined_rain, width=12, height=16, units="in")

# PCA
pca_res <- prcomp(all_libs[desc_cols], center=TRUE, scale.=TRUE)
scores  <- augment(pca_res, all_libs)
cent <- scores %>%
  group_by(Library) %>%
  summarise(PC1=mean(.fittedPC1), PC2=mean(.fittedPC2))

p_pca_shift <- ggplot(scores, aes(.fittedPC1, .fittedPC2, color=Library)) +
  geom_point(alpha=0.3, size=1.2) +
  geom_point(data=cent, aes(PC1, PC2), shape=18, size=5) +
  geom_path(data=cent, aes(PC1,PC2), arrow=arrow(length=unit(0.25,"cm")), color="black") +
  stat_ellipse(type="norm", level=0.95) +
  theme_minimal(base_family="sans", base_size=12) +
  labs(title="PCA: Ortho, Meta, Para, Dihaloacetone",
    x=sprintf("PC1 (%.0f%%)",100*summary(pca_res)$importance[2,1]),
    y=sprintf("PC2 (%.0f%%)",100*summary(pca_res)$importance[2,2])) +
  scale_color_manual(values = lib_colors) +
  theme(legend.position="right",
     legend.title=element_text(size=10, family="sans"),
     legend.text =element_text(size=9, family="sans"))
print(p_pca_shift)
ggsave("Linker_Diversity_pca_shift.svg", p_pca_shift, width=14, height=10, units="in")

# Legend-only plot
legend_only <- ggplot(all_libs, aes(x = Total.Molweight, fill = Library)) +
  geom_histogram(alpha = 0.6, binwidth = 40) +
  scale_fill_manual(values = lib_colors) +
  theme_void() +
  theme(legend.position = "center",
    legend.title = element_text(size = 10, family = "sans"),
    legend.text  = element_text(size = 9, family = "sans"),
    legend.key.size = unit(0.5, "cm"),
    legend.direction = "horizontal") +
  guides(fill = guide_legend(title = "Library"))

legend_grob <- get_legend(legend_only)
print(legend_grob)
ggsave("Linker_Diversity_histogram_legend.svg", legend_grob, width=3, height=1, units="in")
