# This script downloads and preprocesses gene-level data and metadata from
# all samples in recount3
#
# Output:
# A tsv file containing an array of gene expression data, and a metadata file

# R version used: 4.1.0

# Some conda issue causes the installation of GenomeInfoDbData to fail silently, preventing
# recount3 from being installed. I recommend running this script from a base r installation 
# rather than a conda env

#install.packages("BiocManager", repos = "https://cran.r-project.org", dependencies=TRUE)
#install.packages("dplyr", repos = "https://cran.r-project.org", dependencies=TRUE)
#install.packages("tibble", repos = "https://cran.r-project.org", dependencies=TRUE)
#install.packages("tidyr", repos = "https://cran.r-project.org", dependencies=TRUE)
#BiocManager::install()
#BiocManager::install("recount3")

library(recount3)
library(dplyr)
library(tibble)
library(tidyr)

getMetadata <- function(project, project_home, metadata_df) {
  # Append the metadata for a project to a dataframe
  #
  # Args:
  #  project: The accession for the project whose metadata is to be added
  #  project_home: The subdirectory the project is stored in within recount3
  #  metadata_df: The accumulated metadata so far, or NULL
  #
  # Returns:
  #  new_df - A dataframe with the metadata from the project appended
  
  # These are global variables because I can't figure out R scoping for the life of me
  # I think tryCatch creates a new function which is causing variables initialized in the
  # getMetadata scope to be invisible to the inner function scope. Probably more reasonable
  # than python scoping, but I'm not sure what else to do if I want a loop control variable that 
  # is visible in both the outer and inner scope
  read_successful <<- FALSE
  reads_attempted <<- 0
  
  while (!read_successful & reads_attempted < 5){
    result <- tryCatch(
      {
        result <- metadata_df
        reads_attempted <- reads_attempted + 1
        url <- recount3::locate_url(project=project,
                                    project_home = project_home,
                                    organism='human',
                                    type='metadata')
        
        metadata_files <- recount3::file_retrieve(url)
        metadata <- recount3::read_metadata(metadata_files)
        
        if (is.null(metadata_df)) {
          result <- metadata
        } else if (nrow(metadata) == 0){
          # This keeps the script from looking at the cache repeatedly for studies without metadata
          metadata[nrow(metadata)+1,] <- NA
          metadata$study <- project
          result <- rbind(metadata_df, metadata)
        } else {
          result <- rbind(metadata_df, metadata)
        }
        
        read_successful <- TRUE
        return(result)
      },
      error = function(cond) {
        print(cond)
        print('read failed, trying again...')
        Sys.sleep(2 ^ reads_attempted)
        return(NULL)
      }
    )
  }
  if (is.null(result)) {
    return(metadata_df)
  }
  return(result)
}

getCounts <- function(project, project_home) {
  # Append the count data for a project to a dataframe
  #
  # Args:
  #  project: The accession for the project whose read counts are to be added
  #  project_home: The subdirectory the project is stored in within recount3
  #
  # Returns:
  #  counts - A dataframe where the rows are genes and the columns are samples

  read_successful <<- FALSE
  reads_attempted <<- 0
  
  while (!read_successful & reads_attempted < 5){
    counts <- tryCatch(
      {
        reads_attempted <- reads_attempted + 1
        url <- recount3::locate_url(project=project,
                                    project_home = project_home,
                                    organism='human',
                                    type='gene')
        
        count_files <- recount3::file_retrieve(url)
        counts <- recount3::read_counts(count_files)
        
        read_successful <- TRUE
        return(counts)
      },
      error = function(cond) {
        print(cond)
        print('read failed, trying again...')
        Sys.sleep(2 ^ reads_attempted)
        return(NULL)
      }
    )
  }
  return(counts)
}


# Make sure R's working directory is in the correct spot to make relative paths resolve correctly
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

data.dir <- file.path("..","data", "recount3")
dir.create(data.dir, recursive = TRUE, showWarnings = FALSE)

# Create a cache for storing data
recount3::recount3_cache()

# Pull down information about all the samples
samples <- recount3::available_samples(organism='human')

projects <- dplyr::distinct(dplyr::select(samples, project, project_home))

metadata_path <- '../data/metadata_df.rda'
metadata_df <- NULL

if(file.exists(metadata_path)){
  # Restores `metadata_df`
  load(metadata_path)
}

# There has to be a cleaner way to do this, but I am an R novice and "apply a function
# that returns a variable number of rows" doesn't seem to be a common use-case.
# The network connection/disk io for file_retrieve is probably the bottleneck anyway
# so I'm not too worried.
for(i in seq_len(nrow(projects))){
  print(i)
  row <- projects[i,]
  project <- row$project
  
  if (project %in% metadata_df$study) {
    print(paste('already processed ', project))
    next
  }
  
  home <- row$project_home

  # Get only sra data
  if (home != "data_sources/sra") {
    print(paste('Skipping ', project, 'from', home))
    next
  }
  
  print(project)
  metadata_df <- getMetadata(project, home, metadata_df)
  
  if (i %% 10 == 0) {
    save(metadata_df, file=metadata_path)
  }
}
save(metadata_df, file=metadata_path)

# This could be in the same loop as aquiring metadata, 
# but I think it makes sense to be able to run it separately
out_file <- '../data/sra_counts.tsv'
processed_out <- '../data/projects_processed.rda'
projects_processed <- c()
if(file.exists(processed_out)){
  # Restores `projects_processed`
  load(processed_out)
}

for(i in seq_len(nrow(projects))){
  print(i)
  row <- projects[i,]
  project <- row$project
  
  if (project %in% projects_processed) {
    print(paste('already processed ', project))
    next
  }
  
  home <- row$project_home
  
  # Get only sra data
  if (home != "data_sources/sra") {
    print(paste('Skipping ', project, 'from', home))
    next
  }
  
  print(project)
  counts <- getCounts(project, home)
  
  # The dataset won't fit into memory, so we need to write it to disk
  # As a result, we need samples to be rows, not columns
  counts <- t(counts)
  
  # write header with data
  if (i == 1){
    write.table(counts, file=out_file, col.names=TRUE, sep='\t', )
  } else {
  	# Write counts to file
  	write.table(counts, file=out_file, append=TRUE, sep='\t', col.names=FALSE)
  }
  
  projects_processed <- c(projects_processed, project)
  # I'm a little more careful about duplication here because 
  # it's much harder to deduplicate data that doesn't fit in memory
  save(projects_processed, file=processed_out)
}
