import itertools

DATASETS, = glob_wildcards("dataset_configs/{dataset}.yml")

SUPERVISED, = glob_wildcards("model_configs/supervised/{supervised}.yml")
IMPUTE, = glob_wildcards("model_configs/imputation/{impute}.yml")
UNSUPERVISED, = glob_wildcards("model_configs/unsupervised/{unsupervised}.yml")
SEMISUPERVISED, = glob_wildcards("model_configs/semi-supervised/{semisupervised}.yml")

NUM_SEEDS = 3

#top_five_tissues = ['Blood', 'Breast', 'Stem_Cell', 'Cervix', 'Brain']
top_five_tissues = ['Blood', 'Breast', 'Stem_Cell']

combo_iterator = itertools.combinations(top_five_tissues, 2)
TISSUE_STRING = ['.'.join(pair) for pair in combo_iterator]


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
        "data/no_scrna_tpm.pkl",
        # Binary classification
        expand("results/{tissues}.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               tissues=TISSUE_STRING,
               ),
        # Binary classification study corrected
        expand("results/{tissues}.{supervised}_{seed}-signal_removed.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               tissues=TISSUE_STRING,
               ),
        expand("results/{tissues}.{supervised}_{seed}-study_corrected.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               tissues=TISSUE_STRING,
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
        # sample_split sex prediction
        expand("results/sample-split-sex-prediction.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # study_split sex prediction
        expand("results/study-split-sex-prediction.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # sample_split signal removed
        expand("results/sample-split-signal-removed.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # study_split signal removed
        expand("results/study-split-signal-removed.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # sample_split be_corrected
        expand("results/sample-split-study-corrected.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # study_split be_corrected
        expand("results/study-split-study-corrected.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # tissue_split
        expand("results/tissue-split.{supervised}_{seed}.tsv",
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
        "data/no_scrna_tpm.tsv "
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
    shell:
        "python saged/pickle_tsv.py data/no_scrna_tpm.tsv data/no_scrna_tpm.pkl"

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
        "--weighted_loss "

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
        "--weighted_loss "

rule tissue_prediction_signal_removed:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/{tissue1}.{tissue2}.{supervised}_{seed}-signal_removed.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}-signal_removed.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--signal_removal "

rule tissue_prediction_study_corrected:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/{tissue1}.{tissue2}.{supervised}_{seed}-study_corrected.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}-study_corrected.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--study_correct "

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
        "--weighted_loss "

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
        "--weighted_loss "

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
        "--biobert "
        "--weighted_loss "

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
        "--weighted_loss "

rule sample_level_control_sex_prediction:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        "data/combined_human_mouse_meta_v2.csv",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/sample-split-sex-prediction.{supervised}_{seed}.tsv"
    shell:
        "python saged/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/sample-split-sex-prediction.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--sample_split "
        "--weighted_loss "
        "--use_sex_labels "

rule sample_level_control_signal_removed:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/sample-split-signal-removed.{supervised}_{seed}.tsv"
    shell:
        "python saged/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/sample-split-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--sample_split "
        "--weighted_loss "
        "--signal_removal "

rule sample_level_be_corrected:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/sample-split-study-corrected.{supervised}_{seed}.tsv"
    shell:
        "python saged/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/sample-split-study-corrected.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--sample_split "
        "--weighted_loss "
        "--study_correct  "

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
        "--weighted_loss "

rule study_level_sex_prediction:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        "data/combined_human_mouse_meta_v2.csv",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/study-split-sex-prediction.{supervised}_{seed}.tsv"
    shell:
        "python saged/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/study-split-sex-prediction.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--weighted_loss "
        "--use_sex_labels "

rule study_level_signal_removed:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/study-split-signal-removed.{supervised}_{seed}.tsv"
    shell:
        "python saged/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/study-split-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--weighted_loss "
        "--signal_removal "

rule study_level_be_corrected:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/study-split-study-corrected.{supervised}_{seed}.tsv"
    shell:
        "python saged/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/study-split-study-corrected.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--weighted_loss "
        "--study_correct  "

rule tissue_split:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/tissue-split.{supervised}_{seed}.tsv"
    shell:
        "python saged/tissue_split.py {input.dataset_config} {input.supervised_model} "
        "results/tissue-split.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--weighted_loss "
