import itertools

DATASETS, = glob_wildcards("dataset_configs/{dataset}.yml")

SUPERVISED, = glob_wildcards("model_configs/supervised/{supervised}.yml")
IMPUTE, = glob_wildcards("model_configs/imputation/{impute}.yml")
UNSUPERVISED, = glob_wildcards("model_configs/unsupervised/{unsupervised}.yml")
SEMISUPERVISED, = glob_wildcards("model_configs/semi-supervised/{semisupervised}.yml")

NUM_SEEDS = 3

recount_top_five_tissues = ['Blood', 'Breast', 'Stem_Cell', 'Cervix', 'Brain']
combo_iterator = itertools.combinations(recount_top_five_tissues, 2)
RECOUNT_TISSUE_STRING = ['.'.join(pair) for pair in combo_iterator]

gtex_top_five_tissues = ['Blood', 'Brain', 'Skin', 'Esophagus', 'Blood_Vessel']
combo_iterator = itertools.combinations(gtex_top_five_tissues, 2)
GTEX_TISSUE_STRING = ['.'.join(pair) for pair in combo_iterator]

wildcard_constraints:
    # Random seeds should be numbers
    seed="\d+"

rule all:
    input:
        "data/recount_text.txt",
        "data/recount_embeddings.hdf5",
        "data/recount_metadata.tsv",
        "data/manifest.tsv",
        "data/no_scrna_counts.tsv",
        "data/gene_lengths.tsv",
        "data/no_scrna_tpm.tsv",
        "data/no_scrna_tpm.pkl",
        "data/gtex_tpm.gct",
        "data/gtex_sample_attributes.txt",
        "data/gtex_normalized.tsv",
        "data/gtex_normalized.pkl",
        "data/batch_sim_data.tsv",
        "data/linear_batch_sim_data.tsv",
        "data/no_signal_batch_sim_data.tsv",
        # Recount Binary classification
        expand("results/{tissues}.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               tissues=RECOUNT_TISSUE_STRING,
               ),
        # GTEx Binary classification
        expand("results/gtex.{tissues}.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               tissues=GTEX_TISSUE_STRING,
               ),
        expand("results/gtex-signal-removed.{tissues}.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               tissues=GTEX_TISSUE_STRING,
               ),
        # Binary classification w/ corrections
        expand("results/{tissues}.{supervised}_{seed}-signal_removed.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               tissues=RECOUNT_TISSUE_STRING,
               ),
        expand("results/{tissues}.{supervised}_{seed}-split_signal.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               tissues=RECOUNT_TISSUE_STRING,
               ),
        expand("results/{tissues}.{supervised}_{seed}-signal_removed_sample_level.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               tissues=RECOUNT_TISSUE_STRING,
               ),
        expand("results/{tissues}.{supervised}_{seed}-study_corrected.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               tissues=RECOUNT_TISSUE_STRING,
               ),
        # Multi-tissue prediction
        expand("results/all-tissue.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # Multi-tissue prediction sample split
        expand("results/all-tissue_sample-split.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # Multi-tissue prediction be_corrected
        expand("results/all-tissue.{supervised}_{seed}_be_corrected.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        expand("results/all-tissue.{supervised}_{seed}_signal_removed.tsv",
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
        # Multi-tissue prediction
        expand("results/gtex-all-tissue.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # Multi-tissue prediction
        expand("results/gtex-all-tissue-signal-removed.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # Sim results
        expand("results/sim-data.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # Sim results be corrected
        expand("results/sim-data-signal-removed.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # Linear sim results
        expand("results/linear-sim-data.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        expand("results/linear-sim-data-signal-removed.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        expand("results/linear-sim-data-split-signal.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # No signal sim
        expand("results/no-signal-sim-data.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        expand("results/no-signal-sim-data-signal-removed.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        expand("results/no-signal-sim-data-split-signal.{supervised}_{seed}.tsv",
               supervised=SUPERVISED,
               seed=range(0,NUM_SEEDS),
               ),
        # Split-signal sim
        expand("results/sim-data-split-signal.{supervised}_{seed}.tsv",
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

rule download_manifest:
    output:
        "data/manifest.tsv"
    shell:
        "bash saged/download_manifest.sh"

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
    threads: 4
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    wildcard_constraints:
        tissue1='[a-zA-Z]+_?[a-zA-Z]*',
        tissue2='[a-zA-Z]+_?[a-zA-Z]*'
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--disable_optuna "

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
        "--disable_optuna "

rule all_tissue_sample_split:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/all-tissue_sample-split.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/all-tissue_sample-split.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--sample_split "
        "--disable_optuna "

rule tissue_prediction_signal_removed:
    threads: 4
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
        "--correction signal "
        "--disable_optuna "

rule tissue_prediction_split_signal_removal:
    threads: 4
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/{tissue1}.{tissue2}.{supervised}_{seed}-split_signal.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}-split_signal.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--correction split_signal "
        "--disable_optuna "

rule tissue_prediction_signal_removed_sample_split:
    threads: 4
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/{tissue1}.{tissue2}.{supervised}_{seed}-signal_removed_sample_level.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}-signal_removed_sample_level.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--correction signal "
        "--sample_split "
        "--disable_optuna "

rule tissue_prediction_study_corrected:
    threads: 4
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
        "--correction study "
        "--disable_optuna "

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
        "--correction study "
        "--weighted_loss "
        "--disable_optuna "

rule all_tissue_prediction_signal_removed:
    threads: 8
    input:
        "dataset_configs/recount_dataset.yml",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/all-tissue.{supervised}_{seed}_signal_removed.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/all-tissue.{wildcards.supervised}_{wildcards.seed}_signal_removed.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--correction signal "
        "--weighted_loss "
        "--disable_optuna "

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
        "--correction signal "

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
        "--correction study  "

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
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/study-split-sex-prediction.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--weighted_loss "
        "--use_sex_labels "
        "--disable_optuna "

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
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/sample-split-sex-prediction.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--sample_split "
        "--weighted_loss "
        "--use_sex_labels "
        "--disable_optuna "

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
        "--correction signal "

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
        "--correction study  "

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

rule download_gtex:
    output:
        "data/gtex_tpm.gct",
        "data/gtex_sample_attributes.txt"
    shell:
        "bash saged/download_gtex_data.sh"

rule preprocess_gtex:
    input:
        "data/gtex_tpm.gct"
    output:
        "data/gtex_normalized.tsv"
    shell:
        "python saged/normalize_gtex.py data/gtex_tpm.gct data/gtex_normalized.tsv"

rule pickle_gtex:
    input:
        "data/gtex_normalized.tsv"
    output:
        "data/gtex_normalized.pkl"
    shell:
        "python saged/pickle_tsv.py data/gtex_normalized.tsv data/gtex_normalized.pkl"

rule all_tissue_gtex:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex-all-tissue.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/gtex-all-tissue.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset gtex"

rule all_tissue_signal_removed_gtex:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex-all-tissue-signal-removed.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/gtex-all-tissue-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset gtex "
        "--correction signal "

rule gtex_binary_prediction:
    threads: 4
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex.{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/gtex.{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset gtex "

rule gtex_binary_prediction_signal_removed:
    threads: 4
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex-signal-removed.{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/gtex-signal-removed.{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset gtex "
        "--correction signal "

rule simulate_data:
    threads: 8
    output:
        "data/batch_sim_data.tsv"
    shell:
        "python saged/simulate_be_data.py {output}"

rule simulate_linear_data:
    threads: 8
    output:
        "data/linear_batch_sim_data.tsv"
    shell:
        "python saged/simulate_be_data.py {output} --n_nonlinear 0"

rule simulate_no_signal_data:
    threads: 8
    output:
        "data/no_signal_batch_sim_data.tsv"
    shell:
        "python saged/simulate_be_data.py {output} --n_linear 0 --n_nonlinear 0 --n_random 2500"

rule sim_prediction:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/sim_dataset.yml",
        data="data/batch_sim_data.tsv"
    output:
        "results/sim-data.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/sim-data.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset sim "

rule sim_prediction_signal_removed:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/sim_dataset.yml",
        data="data/batch_sim_data.tsv"
    output:
        "results/sim-data-signal-removed.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/sim-data-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--correction signal "
        "--dataset sim "

rule linear_sim_prediction:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/linear_sim_dataset.yml",
        data="data/linear_batch_sim_data.tsv"
    output:
        "results/linear-sim-data.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/linear-sim-data.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset sim"

rule linear_sim_prediction_signal_removed:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/linear_sim_dataset.yml",
        data="data/linear_batch_sim_data.tsv"
    output:
        "results/linear-sim-data-signal-removed.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/linear-sim-data-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--correction signal "
        "--dataset sim"

rule linear_sim_prediction_split_signal:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/linear_sim_dataset.yml",
        data="data/linear_batch_sim_data.tsv"
    output:
        "results/linear-sim-data-split-signal.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/linear-sim-data-split-signal.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--correction split_signal "
        "--dataset sim"

rule no_signal_sim_prediction:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/no_signal_sim_dataset.yml",
        data="data/no_signal_batch_sim_data.tsv"
    output:
        "results/no-signal-sim-data.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/no-signal-sim-data.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset sim "

rule no_signal_sim_prediction_signal_removed:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/no_signal_sim_dataset.yml",
        data="data/no_signal_batch_sim_data.tsv"
    output:
        "results/no-signal-sim-data-signal-removed.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/no-signal-sim-data-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--correction signal "
        "--dataset sim "

rule no_signal_sim_prediction_split:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/no_signal_sim_dataset.yml",
        data="data/no_signal_batch_sim_data.tsv"
    output:
        "results/no-signal-sim-data-split-signal.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/no-signal-sim-data-split-signal.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--correction split_signal "
        "--dataset sim "

rule sim_split_signal:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/sim_dataset.yml",
        data="data/batch_sim_data.tsv"
    output:
        "results/sim-data-split-signal.{supervised}_{seed}.tsv"
    shell:
        "python saged/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/sim-data-split-signal.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset sim "
        "--correction split_signal "