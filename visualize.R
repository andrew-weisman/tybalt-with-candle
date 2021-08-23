#!/usr/bin/env Rscript

# Pan-Cancer Variational Autoencoder
# Gregory Way 2017
#
# Visualize t-SNE plots of VAE, ADAGE, and raw RNAseq data features

library(dplyr)
library(ggplot2)
library(readr)
library(getopt)

library(heatmap.plus)

cmd_args   = commandArgs();

cmd_options= commandArgs(TRUE)

usage <- "\nUsage: tybalt_visualize.R <data.tsv> [ options (-h to list) ] \n"
nargs = length(cmd_args)

options_vector <-
  c('tsne_adage',           'a',0,"logical","Display tSNE ADAGE   resulta",
    'tsne_vae',             'v',0,"logical","Display tSNE VAE     resulta",
    'tsne_rnaseq',          'r',0,"logical","Display tSNE RNA-seq resulta",
    'gender_encodings',     'g',0,"logical","Display gender encodings",
    'sample_type_encodings','t',0,"logical","Display sample_type activations", 
    'heat_map',             'm',0,"logical","Display Heat Map",
    'help',                 'h',0,"logical","Show a detailed help message"
   )
command_spec <- matrix(options_vector,  ncol=5, byrow=TRUE)
option_inds  <- which(substr(cmd_options,1,1) == '-')

if (length(option_inds) > 0) {
  first_option_ind <- ifelse(length(option_inds) > 0, min(option_inds), Inf)
  first_option_ind <- min(first_option_ind, length(cmd_options)+1)   # 
  opt <- getopt(command_spec, opt = cmd_options[first_option_ind:length(cmd_options)])
} else {
  first_option_ind <- -1
  opt <- list()
}

# Setting default options
if ( is.null(opt$tsne_adage            ) ) { opt$tsne_adage            = FALSE }
if ( is.null(opt$tsne_vae              ) ) { opt$tsne_vae              = FALSE }
if ( is.null(opt$tsne_rnaseq           ) ) { opt$tsne_rnaseq           = FALSE }
if ( is.null(opt$gender_encodings      ) ) { opt$gender_encodings      = FALSE }
if ( is.null(opt$sample_type_encodings ) ) { opt$sample_type_encodings = FALSE }
if ( is.null(opt$heat_map              ) ) { opt$heat_map              = FALSE }
if ( is.null(opt$help                  ) ) { opt$help                  = FALSE }

plot_tsne <- function(tsne_df, color_df) {
  # Function to output tsne visualizations colored by cancer-type for the given
  # input data (compressed by either Tybalt or ADAGE or raw RNAseq data)
  p <- ggplot(tsne_df, aes(x = `1`, y = `2`, color = acronym)) +
    geom_point(size = 0.001) +
    scale_colour_manual(limits = color_df$`Study Abbreviation`,
                        values = color_df$`Hex Colors`,
                        na.value = "black",
                        breaks = palette_order) +
    theme_classic() +
    theme(legend.title = element_text(size = 8),
          legend.text = element_text(size = 5),
          legend.key.size = unit(0.5, "line")) +
    guides(colour = guide_legend(override.aes = list(size = 0.1),
                                 title = 'Cancer-Type'))
  return(p)
}

plotActivation <- function(df, x, y, covariate, legend_show = TRUE) {
  # Helper function to plot given coordinates and color by specific covariates
  #
  # Arguments:
  # df - dataframe for plotting, must include x, y, and covariate variables
  # x - encoding integer or string to be plotted on x axis
  # y - encoding integer or string to be plotted on y axis
  # covariate - which factors to distinguish by color in the plot
  # legend_show - boolean to determine if the plot should include a legend
  #
  # Output:
  # Will return the constructed ggplot object

  x_coord <- paste(x)
  y_coord <- paste(y)
  color_ <- covariate

  p <- ggplot(df, aes_string(x = df[[x_coord]], y = df[[y_coord]],
                           color = color_)) +
    geom_point() +
    xlab(paste("encoding", x_coord)) +
    ylab(paste("encoding", y_coord)) +
    theme_bw() +
    theme(text = element_text(size = 20))

  if (color_ == "acronym") {
    p <- p + scale_colour_manual(limits = tcga_colors$`Study Abbreviation`,
                                 values = tcga_colors$`Hex Colors`,
                                 na.value = "black",
                                 breaks = palette_order)
  }

  if (!legend_show) {
    p <- p + theme(legend.position = "none")
  }
  return(p)
}

# Produce combined clinical data file if it does not exist
comb_out_file <- file.path("data", "tybalt_features_with_clinical.tsv")
if (!file.exists(comb_out_file)){
    print("Producing a combined clinical data file")
    vae_file <- file.path("data", "encoded_rnaseq_onehidden_warmup_batchnorm.tsv")
    clinical_file <- file.path("data", "clinical_data.tsv")
    vae_df <- readr::read_tsv(vae_file)
    colnames(vae_df)[1] <- "sample_id"
    clinical_df <- readr::read_tsv(clinical_file)

    vae_df <- vae_df %>%
        dplyr::rowwise() %>%
        dplyr::mutate(sample_base = substring(sample_id, 1, 12))

    combined_df <- dplyr::inner_join(vae_df, clinical_df,
                                 by = c("sample_base" = "sample_id"))
    combined_df <- combined_df[!duplicated(combined_df$sample_id), ]

    # Process improper drug names that cause parsing failures
    combined_df$drug <- tolower(combined_df$drug )
    combined_df$drug <- gsub("\t", "", combined_df$drug)
    combined_df$drug <- gsub('"', "", combined_df$drug)
    combined_df$drug <- gsub("\\\\", "", combined_df$drug)
    write.table(combined_df, file = comb_out_file, sep = "\t", row.names = FALSE)
}

# Produce file data/tcga_colors.tsv
if (!file.exists(file.path("data", "tcga_colors.tsv"))){
    tcga_colors_file <- paste(Sys.getenv("TYBALT_HOME"), "data", 
                              "tcga_colors.tsv", sep = "/")
    file.copy(tcga_colors_file, "data")
}

# Load encodings file with matched clinical data to subset
clinical_file <- file.path("data", "tybalt_features_with_clinical.tsv")
clinical_df <- readr::read_tsv(clinical_file)
clinical_df <- clinical_df %>% dplyr::select(sample_id, acronym)

# Load color data
tcga_colors <- readr::read_tsv(file.path("data", "tcga_colors.tsv"))
tcga_colors <- tcga_colors[order(tcga_colors$`Study Abbreviation`), ]

palette_order <- c("BRCA", "PRAD", "TGCT", "KICH", "KIRP", "KIRC", "BLCA",
                   "OV", "UCS", "CESC", "UCEC", "THCA", "PCPG", "ACC", "SKCM",
                   "UVM", "HNSC", "SARC", "ESCA", "STAD", "COAD", "READ",
                   "CHOL", "PAAD", "LIHC", "DLBC", "MESO", "LUSC", "LUAD",
                   "GBM", "LGG", "LAML", "THYM", NA)

# Plot and save VAE tsne
if (opt$tsne_vae) {
    if (!file.exists(file.path("results", "vae_tsne_features.tsv"))){
        cat("File results/vae_tsne_features.tsv does not exist\n")
        cat("Process VAE data first\n")
        quit("yes")
    }

    # Load tsne data
    tsne_vae_file <- file.path("results", "vae_tsne_features.tsv")
    tsne_vae_df <- readr::read_tsv(tsne_vae_file)
    tsne_vae_df <- dplyr::inner_join(tsne_vae_df, clinical_df,
                                     by = c("tcga_id" = "sample_id"))

    vae_tsne_pdf_out_file <- file.path("figures", "tsne_vae.pdf")
    vae_tsne_png_out_file <- file.path("figures", "tsne_vae.png")
    p <- plot_tsne(tsne_vae_df, tcga_colors)
    ggsave(vae_tsne_pdf_out_file, plot = p, width = 6, height = 4.5)
    ggsave(vae_tsne_png_out_file, plot = p, width = 4, height = 3)  # PNG for repo

    X11()
    plot(p)
    message("Press Return To Continue")
    invisible(readLines("stdin", n=1))
}

# Plot and save ADAGE tsne
if (opt$tsne_adage) {
    if (!file.exists(file.path("results", "adage_tsne_features.tsv"))){
        cat("File results/adage_tsne_features.tsv does not exist\n")
        cat("Process ADAGE data first\n")
        quit("yes")
    }

    # Load tsne data
    tsne_adage_file <- file.path("results", "adage_tsne_features.tsv")
    tsne_adage_df <- readr::read_tsv(tsne_adage_file)
    tsne_adage_df <- dplyr::inner_join(tsne_adage_df, clinical_df,
                                       by = c("tcga_id" = "sample_id"))

    adage_tsne_pdf_out_file <- file.path("figures", "tsne_adage.pdf")
    adage_tsne_png_out_file <- file.path("figures", "tsne_adage.png")
    p <- plot_tsne(tsne_adage_df, tcga_colors)
    ggsave(adage_tsne_pdf_out_file, plot = p, width = 6, height = 4.5)
    ggsave(adage_tsne_png_out_file, plot = p, width = 4, height = 3)  

    X11()
    plot(p)
    message("Press Return To Continue")
    invisible(readLines("stdin", n=1))
}

# Plot and save RNAseq tsne
if (opt$tsne_rnaseq) {
        if (!file.exists(file.path("results", "rnaseq_tsne_features.tsv"))){
        cat("File results/rnaseq_tsne_features.tsv does not exist\n")
        cat("Process RNA-seq data first\n")
        quit("yes")
    }

    # Load tsne data
    tsne_rnaseq_file <- file.path("results", "rnaseq_tsne_features.tsv")
    tsne_rnaseq_df <- readr::read_tsv(tsne_rnaseq_file)
    tsne_rnaseq_df <- dplyr::inner_join(tsne_rnaseq_df, clinical_df,
                                    by = c("tcga_id" = "sample_id"))

    rnaseq_tsne_pdf_out_file <- file.path("figures", "tsne_rnaseq.pdf")
    rnaseq_tsne_png_out_file <- file.path("figures", "tsne_rnaseq.png")
    p <- plot_tsne(tsne_rnaseq_df, tcga_colors)
    ggsave(rnaseq_tsne_pdf_out_file, plot = p, width = 6, height = 4.5)
    ggsave(rnaseq_tsne_png_out_file, plot = p, width = 4, height = 3)

    X11()
    plot(p)
    message("Press Return To Continue")
    invisible(readLines("stdin", n=1))
}

# Plot heat map
if (opt$gender_encodings | opt$sample_type_encodings | opt$heat_map)
{
  vae_clinical_file <- file.path("data", "tybalt_features_with_clinical.tsv")
  combined_df <- readr::read_tsv(vae_clinical_file)

  # Load official colors
  tcga_colors <- readr::read_tsv(file.path("data", "tcga_colors.tsv"))

  tcga_colors <- tcga_colors[order(tcga_colors$`Study Abbreviation`), ]
  match_colors <- match(combined_df$acronym, tcga_colors$`Study Abbreviation`)
  combined_df$colors <- tcga_colors$`Hex Colors`[match_colors]

  if (opt$gender_encodings) {
    gender_encodings        <- file.path("figures", "gender_encodings.png")
    gender_encodings_legend <- file.path("figures", "gender_encodings_legend.png")

    # Output plots with and without legend
    gender_fig        <- plotActivation(combined_df, x = 82, y = 85,
                                        covariate = "gender",legend_show = FALSE)
    gender_fig_legend <- plotActivation(combined_df, x = 82, y = 85,
                                        covariate = "gender",legend_show = TRUE)
    ggsave(gender_encodings, plot = gender_fig, width = 6, height = 5)
    ggsave(gender_encodings_legend, plot = gender_fig_legend, width = 6, height = 5)

    X11()
    plot(gender_fig_legend)
    message("Press Return To Continue")
    invisible(readLines("stdin", n=1))
  }

  if (opt$sample_type_encodings) {
    sample_type_file      <- file.path("figures", "sample_type_encodings.png")
    sample_legend_file    <- file.path("figures", "sample_type_encodings_legend.png")
    # Subset new dataframe for visualizing Melanoma (SKCM) activation. We
    # previously observed the two encodings (53 and 66) separated SKCM tumors.
    met_df <- combined_df %>%
    dplyr::select("53", "66", "sample_type", "acronym") %>%
    dplyr::mutate(label = paste0(sample_type, acronym))

    # Change labels to capture melanoma vs. non-melanoma
    met_df$label[(met_df$acronym == "SKCM") &
                 (met_df$sample_type != "Metastatic")] <- "Non-metastatic SKCM"
    met_df$label[(met_df$acronym != "SKCM") &
                 (met_df$sample_type != "Metastatic")] <- "Non-metastatic Other"
    met_df$label[(met_df$acronym != "SKCM") &
               (met_df$sample_type == "Metastatic")] <- "Metastatic Other"
    met_df$label[(met_df$label == "MetastaticSKCM")] <- "Metastatic SKCM"

    p <- ggplot(met_df, aes(x = `53`, y = `66`, color = label)) +
              geom_point() +
              xlab(paste("encoding 53")) +
              ylab(paste("encoding 66")) +
              theme_bw() +
              theme(text = element_text(size = 20))

    # Save plots with and without legend
    p_legend <- p + theme(legend.position = "none")

    ggsave(sample_type_file,   plot = p, width = 6, height = 5)
    ggsave(sample_legend_file, plot = p_legend, width = 6, height = 5)
     
    X11()
    plot(p)                
    message("Press Return To Continue")
    invisible(readLines("stdin", n=1))
  }

  # Plot heat map
  if (opt$heat_map) {

    encodings_df <- combined_df %>% dplyr::select(num_range('', 1:100))
    encodings_matrix <- as.matrix(encodings_df)

    sample_type_vector <- combined_df$sample_type
    sample_type_vector <- sample_type_vector %>%
    dplyr::recode("Primary Tumor" = "green",
                  "Additional - New Primary" = "green",
                  "Recurrent Tumor" = "green",
                  "Metastatic" = "red",
                  "Additional Metastatic" = "red",
                  "Solid Tissue Normal" = "blue",
                  "Primary Blood Derived Cancer - Peripheral Blood" = "purple")
    sex_vector <- combined_df$gender
    sex_vector <- sex_vector %>%
      dplyr::recode("female" = "orange", "male" = "black")

    row_color_matrix <- as.matrix(cbind(sample_type_vector, sex_vector))
    colnames(row_color_matrix) <- c("Sample", "Sex")

    heatmap_pdf_file <- file.path("figures", "encoding_heatmap.pdf")
    X11()
#   pdf(heatmap_pdf_file, width = 8, height = 9)
    heatmap.plus(encodings_matrix, RowSideColors = row_color_matrix,
                 scale = "row", labRow = FALSE, labCol = FALSE,
                 ylab = "Samples", xlab = "VAE Encodings")
    legend(x = -0.08, y = 1.08, xpd = TRUE,
           legend = c("", "Tumor", "Metastasis", "Normal", "Blood Tumor"),
           fill = c("white", "green", "red", "blue", "purple"), border = FALSE,
           bty = "n", cex = 0.7)
    legend(x = 0.05, y = 1.08, xpd = TRUE,
           legend = c("", "Male", "Female"),
           fill = c("white", "black", "orange"), border = FALSE,
           bty = "n", cex = 0.7)
    message("Press Return To Continue")
    invisible(readLines("stdin", n=1))
    dev.off()
  }
}
