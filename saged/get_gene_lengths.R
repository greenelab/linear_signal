#install.packages("BiocManager", repos = "https://cran.r-project.org", dependencies=TRUE)
#install.packages("dplyr", repos = "https://cran.r-project.org", dependencies=TRUE)
#install.packages("tibble", repos = "https://cran.r-project.org", dependencies=TRUE)
#install.packages("tidyr", repos = "https://cran.r-project.org", dependencies=TRUE)
#BiocManager::install()
#BiocManager::install("recount3")
#BiocManager::install("EDASeq")

library(recount3)
library(dplyr)
library(tibble)
library(tidyr)
library (EDASeq)

if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
} else{
  # If running as a script, finding the file is harder
  # https://stackoverflow.com/a/55322344/10930590
  this_file <- commandArgs() %>% 
    tibble::enframe(name = NULL) %>%
    tidyr::separate(col=value, into=c("key", "value"), sep="=", fill='right') %>%
    dplyr::filter(key == "--file") %>%
    dplyr::pull(value)
  
  setwd(dirname(this_file))
}

# Recount3 uses ensembl genes with version numbers, EDASeq expects ensembl genes without the versions
split_gene_name <- function(name){
  return (unlist(strsplit(name, '\\.'))[[1]])
}

# The project doesn't really matter, I'm just using it to get the genes used in recount3
url <- recount3::locate_url(project='SRP103067', project_home='data_sources/sra', type='gene')
count_files <- recount3::file_retrieve(url)
counts <- recount3::read_counts(count_files)

genes <- rownames(counts)
genes_out <- unlist(lapply(genes, split_gene_name))

gene_lengths <- EDASeq::getGeneLengthAndGCContent(genes_out, "hsa")

len_only <- gene_lengths[,1, drop=FALSE]
write.table(len_only, file='../data/gene_lengths.tsv', sep='\t')
