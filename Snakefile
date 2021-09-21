DATASETS, = glob_wildcards("dataset_configs/{dataset}.yml")

SUPERVISED, = glob_wildcards("model_configs/supervised/{supervised}.yml")
IMPUTE, = glob_wildcards("model_configs/imputation/{impute}.yml")
UNSUPERVISED, = glob_wildcards("model_configs/unsupervised/{unsupervised}.yml")
SEMISUPERVISED, = glob_wildcards("model_configs/semi-supervised/{semisupervised}.yml")
NUM_SEEDS = 3

wildcard_constraints:
    # Random seeds should be numbers
    seed="\d+"

rule all:
    input:
        "data/recount_text.txt",
        "data/recount_embeddings.hdf5",
        "data/recount_metadata.tsv",
        "data/no_scrna_counts.tsv",
        "data/gene_lengths.tsv",
        "data/no_scrna_tpm.tsv",
        # TODO add data processing scripts
        # Blood tissue vs breast tissue prediction
        expand("results/Blood.Breast.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # Blood tissue vs breast tissue be corrected
        expand("results/Blood.Breast.{supervised}_{seed}_be_corrected.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # Multi-tissue prediction
        expand("results/all-tissue.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # Multi-tissue prediction be_corrected
        expand("results/all-tissue.{supervised}_{seed}_be_corrected.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # Tissue prediction with imputation pretraining
        expand("results/tissue_impute.{impute}_{seed}.tsv",
               impute=IMPUTE,
               seed=range(0,NUM_SEEDS),
               ),
        # biobert_multitissue
        expand("results/all-tissue-biobert.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # sample_split
        expand("results/sample-split.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # study_split
        expand("results/study-split.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),

rule metadata_to_tsv:
    input:
        "data/metadata_df.rda"
    output:
        "data/recount_metadata.tsv"
    shell:
        "Rscript saged/metadata_to_tsv.R"

rule remove_scrna:
    input:
        "data/sra_counts.tsv",
        "data/recount_metadata.tsv"
    output:
        "data/no_scrna_counts.tsv"
    shell:
        "python saged/remove_scrnaseq.py "
        "data/sra_counts.tsv "
        "data/recount_metadata.tsv "
        "data/no_scrna_counts.tsv "

rule normalize_data:
    input:
        "data/no_scrna_counts.tsv",
        "data/gene_lengths.tsv"
    output:
        "data/no_scrna_tpm.tsv"
    shell:
        "python saged/normalize_recount_data.py "
        "data/no_scrna_counts.tsv "
        "data/gene_lengths.tsv "
        "data/no_scrna_tpm.tsv.tsv "
        "data/recount_metadata.tsv "

rule get_gene_lengths:
    output:
        "data/gene_lengths.tsv"
    shell:
        "Rscript saged/get_gene_lengths.R "

rule pickle_counts:
    input:
        "data/no_scrna_tpm.tsv"
    output:
        "data/no_scrna_tpm.pkl"

rule create_biobert_metadata_file:
    input:
        "data/recount_metadata.tsv"
    output:
        "data/recount_text.txt"
    shell:
        "python saged/extract_metadata_text.py "
        "data/recount_metadata.tsv "
        "data/recount_text.txt "


rule create_biobert_embeddings:
    input:
        "data/recount_text.txt"
    output:
        "data/recount_embeddings.hdf5"
    shell:
        "python biobert-pytorch/embedding/run_embedding.py "
        "--model_name_or_path dmis-lab/biobert-large-cased-v1.1 "
        "--data_path data/recount_text.txt "
        "--output_path data/recount_embeddings.hdf5 "
        "--pooling=sum "
        "--keep_text_order "

rule tissue_prediction:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "

rule all_tissue_prediction:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/all-tissue.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/all-tissue.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "

rule tissue_prediction_be_corrected:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/{tissue1}.{tissue2}.{supervised}_{seed}_be_corrected.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}_be_corrected.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--batch_correction_method limma "

rule all_tissue_prediction_be_corrected:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/all-tissue.{supervised}_{seed}_be_corrected.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/all-tissue.{wildcards.supervised}_{wildcards.seed}_be_corrected.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--batch_correction_method limma "

rule transfer_tissue:
    input:
        "data/recount_tpm.pkl",
        imputation_model = "model_configs/imputation/{impute}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    threads: 8
    output:
        "results/tissue_impute.{impute}_{seed}.tsv"
    shell:
        "python saged/imputation_pretraining.py {input.dataset_config} {input.imputation_model} "
        "results/tissue_impute.{wildcards.impute}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "

rule all_tissue_biobert:
    threads: 16
    input:
        "dataset_configs/recount_dataset.yml",
        "data/recount_embeddings.hdf5",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/all-tissue-biobert.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/all-tissue-biobert.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--biobert"

rule sample_level_control:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/sample-split.{supervised}_{seed}.tsv"
    shell:
        "python saged/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/sample-split.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--sample_split "

rule study_level_control:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/study-split.{supervised}_{seed}.tsv"
    shell:
        "python saged/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/study-split.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "