# This file downloads and unzips GTEx expression data and metadata
curl https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz > data/gtex_tpm.gct.gz
gunzip data/gtex_tpm.gct.gz

curl https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt > data/gtex_sample_attributes.txt