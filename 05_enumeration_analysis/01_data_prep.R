# Add meta library as lib1
lib_meta <- read_csv(
  "C:/Users/Admin/Documents/Hu Lab/DataWarrior/Linker_Enumeration_09112025/34mer_library_products_m_dibromoxylene_linker_09112025.csv",
  show_col_types = FALSE
) %>% mutate(Library = "Meta") %>% rename_with(make.names)

# 01_data_prep.R
# Reads two descriptor CSVs and combines into a single data frame `all_libs`

# 1. Load required packages
library(tidyverse)


# 2. Read & tag each new library by functional group name
lib_trans_1_4_dibromo_2_butene <- read_csv(
  "C:/Users/Admin/Documents/Hu Lab/DataWarrior/Linker_Enumeration_09152025/34mer_library_products_trans_1_4_dibromo_2_butene.csv",
  show_col_types = FALSE
) %>% mutate(Library = "trans-1,4-dibromo-2-butene") %>% rename_with(make.names)

lib_2_3_bis_bromomethyl_quinoxaline <- read_csv(
  "C:/Users/Admin/Documents/Hu Lab/DataWarrior/Linker_Enumeration_09152025/34mer_library_products_2_3_bis_bromomethyl_quinoxaline.csv",
  show_col_types = FALSE
) %>% mutate(Library = "2,3-bis-bromomethyl-quinoxaline") %>% rename_with(make.names)

lib_2_6_bis_bromomethyl_naphthalene <- read_csv(
  "C:/Users/Admin/Documents/Hu Lab/DataWarrior/Linker_Enumeration_09152025/34mer_library_products_2_6_bis_bromomethyl_naphthalene.csv",
  show_col_types = FALSE
) %>% mutate(Library = "2,6-bis-bromomethyl-naphthalene") %>% rename_with(make.names)

lib_3_4_bis_bromomethyl_furan <- read_csv(
  "C:/Users/Admin/Documents/Hu Lab/DataWarrior/Linker_Enumeration_09152025/34mer_library_products_3_4_bis_bromomethyl_furan.csv",
  show_col_types = FALSE
) %>% mutate(Library = "3,4-bis-bromomethyl-furan") %>% rename_with(make.names)

lib_4_6_divinylpyrimidin_2_amine <- read_csv(
  "C:/Users/Admin/Documents/Hu Lab/DataWarrior/Linker_Enumeration_09152025/34mer_library_products_4_6_divinylpyrimidin_2_amine.csv",
  show_col_types = FALSE
) %>% mutate(Library = "4,6-divinylpyrimidin-2-amine") %>% rename_with(make.names)

lib_dibromo_pyridine <- read_csv(
  "C:/Users/Admin/Documents/Hu Lab/DataWarrior/Linker_Enumeration_09152025/34mer_library_products_dibromo_pyridine.csv",
  show_col_types = FALSE
) %>% mutate(Library = "dibromo-pyridine") %>% rename_with(make.names)

lib_divinyl_sulfone <- read_csv(
  "C:/Users/Admin/Documents/Hu Lab/DataWarrior/Linker_Enumeration_09152025/34mer_library_products_divinyl_sulfone.csv",
  show_col_types = FALSE
) %>% mutate(Library = "divinyl-sulfone") %>% rename_with(make.names)

# 3. Combine into one data frame and fix factor ordering
all_libs <- bind_rows(
  lib_meta,
  lib_trans_1_4_dibromo_2_butene,
  lib_2_3_bis_bromomethyl_quinoxaline,
  lib_2_6_bis_bromomethyl_naphthalene,
  lib_3_4_bis_bromomethyl_furan,
  lib_4_6_divinylpyrimidin_2_amine,
  lib_dibromo_pyridine,
  lib_divinyl_sulfone
) %>%
  mutate(
    Library = factor(Library, levels = c(
      "Meta",
      "trans-1,4-dibromo-2-butene",
      "2,3-bis-bromomethyl-quinoxaline",
      "2,6-bis-bromomethyl-naphthalene",
      "3,4-bis-bromomethyl-furan",
      "4,6-divinylpyrimidin-2-amine",
      "dibromo-pyridine",
      "divinyl-sulfone"
    ))
  )

# 4. (optional) Save combined data for downstream scripts
write_csv(all_libs, "all_libs_Enumeration_Linker_Diversity.csv")
