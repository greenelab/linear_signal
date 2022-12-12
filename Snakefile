import itertools

DATASETS, = glob_wildcards("dataset_configs/{dataset}.yml")

SUPERVISED, = glob_wildcards("model_configs/supervised/{supervised}.yml")

NUM_SEEDS = 3

recount_top_five_tissues = ['Blood', 'Breast', 'Stem_Cell', 'Cervix', 'Brain']
combo_iterator = itertools.combinations(recount_top_five_tissues, 2)
RECOUNT_TISSUE_STRING = ['.'.join(pair) for pair in combo_iterator]

gtex_top_five_tissues = ['Blood', 'Brain', 'Skin', 'Esophagus', 'Blood_Vessel']
combo_iterator = itertools.combinations(gtex_top_five_tissues, 2)
GTEX_TISSUE_STRING = ['.'.join(pair) for pair in combo_iterator]

transfer_top_five_tissues = ['Blood', 'Breast', 'Brain', 'Kidney', 'Lung']
combo_iterator = itertools.combinations(transfer_top_five_tissues, 2)
TRANSFER_TISSUE_STRING = ['.'.join(pair) for pair in combo_iterator]

TCGA_GENES = ['EGFR', 'IDH1', 'KRAS', 'PIK3CA', 'SETD2', 'TP53']

wildcard_constraints:
    # Random seeds should be numbers
    seed="\d+"

result_files = [
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
    expand("results/gtex-small.{tissues}.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            tissues=GTEX_TISSUE_STRING,
            ),
    # Recount Transfer Binary classification
    expand("results/recount_transfer.{tissues}.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            tissues=TRANSFER_TISSUE_STRING,
            ),
    # GTEx Transfer Binary classification
    expand("results/gtex_transfer.{tissues}.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            tissues=TRANSFER_TISSUE_STRING,
            ),
    expand("results/recount_transfer.split_signal.{tissues}.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            tissues=TRANSFER_TISSUE_STRING,
            ),
    expand("results/gtex_transfer.split_signal.{tissues}.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            tissues=TRANSFER_TISSUE_STRING,
            ),
    expand("results/gtex-signal-removed.{tissues}.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            tissues=GTEX_TISSUE_STRING,
            ),
    expand("results/gtex-signal-removed-small.{tissues}.{supervised}_{seed}.tsv",
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
    # Multi-tissue prediction
    expand("results/all-tissue.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    # Multi-tissue prediction
    expand("results/recount-transfer.all-tissue.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    # Multi-tissue prediction
    expand("results/gtex-transfer.all-tissue.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    # Signal removed multitissue be correction
    expand("results/recount-transfer.split-signal.all-tissue.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    expand("results/gtex-transfer.split-signal.all-tissue.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    # Multi-tissue prediction sample split
    expand("results/all-tissue_sample-split.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    expand("results/all-tissue.{supervised}_{seed}_signal_removed.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    # pretraining
    expand("results/study-split.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    # study_split sex prediction
    expand("results/study-split-sex-prediction.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    # sex prediction split-signal
    expand("results/sex-prediction-signal-removed.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    # Multi-tissue prediction
    expand("results/gtex-all-tissue.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    expand("results/gtex-all-tissue-small.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    # Multi-tissue prediction
    expand("results/gtex-all-tissue-signal-removed.{supervised}_{seed}.tsv",
            supervised=SUPERVISED,
            seed=range(0,NUM_SEEDS),
            ),
    expand("results/gtex-all-tissue-signal-removed-small.{supervised}_{seed}.tsv",
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
]

figures = [
    'figures/full_signal_combined.svg',
    'figures/simulated_data_combined.svg',
    'figures/signal_removed_combined.svg',
    'figures/recount_binary.svg',
    'figures/no_signal_sim_signal_removed.svg',
    'figures/recount_binary_combined.svg',
    'figures/recount_multiclass_sample_split.svg',
    'figures/recount_pretraining.svg',
]

data_files = [
    "data/sra_counts.tsv",
    "data/metadata_df.rda",
    "data/recount_metadata.tsv",
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
    "data/recount_gtex_genes.tsv",
    "data/recount_gtex_genes.pkl"
]

rule all:
    input:
        result_files,
        figures,
        data_files

rule generate_figures:
    input:
        result_files
    output:
        figures
    shell:
        "jupyter nbconvert --to notebook --execute notebook/analysis/visualize_results.ipynb"


rule metadata_to_tsv:
    input:
        "data/metadata_df.rda"
    output:
        "data/recount_metadata.tsv"
    shell:
        "Rscript saged/metadata_to_tsv.R"

rule download_recount:
    output:
        "data/metadata_df.rda",
        "data/sra_counts.tsv"
    shell:
        "Rscript saged/download_recount3.R "

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
        "python src/remove_scrnaseq.py "
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
        "python src/normalize_recount_data.py "
        "data/no_scrna_counts.tsv "
        "data/gene_lengths.tsv "
        "data/no_scrna_tpm.tsv "
        "data/recount_metadata.tsv "

rule get_gene_lengths:
    output:
        "data/gene_lengths.tsv"
    shell:
        "Rscript saged/get_gene_lengths.R "

rule get_tissue_labels:
    input:
        "data/recount_metadata.tsv",
        "data/no_scrna_tpm.pkl"
    output:
        "data/recount_sample_to_label.pkl",
        "data/no_scrna_tissue_subset.pkl",
    shell:
        "python src/get_tissue_data.py data/no_scrna_tpm.pkl data/recount_metadata.tsv "
        "data/no_scrna_tissue_subset.pkl data/recount_sample_to_label.pkl"

rule pickle_counts:
    input:
        "data/no_scrna_tpm.tsv"
    output:
        "data/no_scrna_tpm.pkl"
    shell:
        "python src/pickle_tsv.py data/no_scrna_tpm.tsv data/no_scrna_tpm.pkl"

rule tissue_prediction:
    threads: 4
    input:
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    wildcard_constraints:
        tissue1='[A-Z][a-z]+_?[A-Z]?[a-z]*',
        tissue2='[A-Z][a-z]+_?[A-Z]?[a-z]*',
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
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
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/all-tissue.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/all-tissue.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "

rule all_tissue_sample_split:
    threads: 8
    input:
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/all-tissue_sample-split.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
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
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/{tissue1}.{tissue2}.{supervised}_{seed}-signal_removed.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}-signal_removed.tsv "
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
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/{tissue1}.{tissue2}.{supervised}_{seed}-signal_removed_sample_level.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}-signal_removed_sample_level.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--correction split_signal "
        "--sample_split "
        "--disable_optuna "

rule tissue_prediction_study_corrected:
    threads: 4
    input:
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/{tissue1}.{tissue2}.{supervised}_{seed}-study_corrected.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
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
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/all-tissue.{supervised}_{seed}_be_corrected.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
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
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/all-tissue.{supervised}_{seed}_signal_removed.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/all-tissue.{wildcards.supervised}_{wildcards.seed}_signal_removed.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--correction split_signal "
        "--weighted_loss "
        "--disable_optuna "

rule sample_level_control:
    threads: 8
    input:
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/sample-split.{supervised}_{seed}.tsv"
    shell:
        "python src/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/sample-split.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--sample_split "
        "--weighted_loss "

rule sample_level_control_signal_removed:
    threads: 8
    input:
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/sample-split-signal-removed.{supervised}_{seed}.tsv"
    shell:
        "python src/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/sample-split-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--sample_split "
        "--weighted_loss "
        "--correction split_signal "

rule sample_level_be_corrected:
    threads: 8
    input:
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/sample-split-study-corrected.{supervised}_{seed}.tsv"
    shell:
        "python src/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/sample-split-study-corrected.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--sample_split "
        "--weighted_loss "
        "--correction study  "

rule study_level_control:
    threads: 8
    input:
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/study-split.{supervised}_{seed}.tsv"
    shell:
        "python src/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/study-split.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--weighted_loss "

rule study_level_sex_prediction:
    threads: 5
    input:
        "data/combined_human_mouse_meta_v2.csv",
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/study-split-sex-prediction.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/study-split-sex-prediction.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--weighted_loss "
        "--use_sex_labels "
        "--disable_optuna "

rule sex_prediction_signal_removed:
    threads: 5
    input:
        "data/combined_human_mouse_meta_v2.csv",
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/sex-prediction-signal-removed.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/sex-prediction-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--weighted_loss "
        "--use_sex_labels "
        "--disable_optuna "
        "--correction split_signal "

rule sample_level_control_sex_prediction:
    threads: 8
    input:
        "data/combined_human_mouse_meta_v2.csv",
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/sample-split-sex-prediction.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
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
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/study-split-signal-removed.{supervised}_{seed}.tsv"
    shell:
        "python src/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/study-split-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--weighted_loss "
        "--correction split_signal "

rule study_level_be_corrected:
    threads: 8
    input:
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/study-split-study-corrected.{supervised}_{seed}.tsv"
    shell:
        "python src/sample_split_control.py {input.dataset_config} {input.supervised_model} "
        "results/study-split-study-corrected.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--weighted_loss "
        "--correction study  "

rule tissue_split:
    threads: 8
    input:
        "data/no_scrna_tpm.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_dataset.yml",
    output:
        "results/tissue-split.{supervised}_{seed}.tsv"
    shell:
        "python src/tissue_split.py {input.dataset_config} {input.supervised_model} "
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
        "python src/normalize_gtex.py data/gtex_tpm.gct data/gtex_normalized.tsv"

rule pickle_gtex:
    input:
        "data/gtex_normalized.tsv"
    output:
        "data/gtex_normalized.pkl"
    shell:
        "python src/pickle_tsv.py data/gtex_normalized.tsv data/gtex_normalized.pkl"

rule all_tissue_gtex:
    threads: 5
    input:
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex-all-tissue.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/gtex-all-tissue.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset gtex"

rule all_tissue_gtex_small:
    threads: 5
    input:
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex-all-tissue-small.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/gtex-all-tissue-small.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset gtex "
        "--range_size .01 "

rule all_tissue_signal_removed_gtex:
    threads: 8
    input:
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex-all-tissue-signal-removed.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/gtex-all-tissue-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset gtex "
        "--correction split_signal "

rule all_tissue_signal_removed_gtex_small:
    threads: 8
    input:
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex-all-tissue-signal-removed-small.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/gtex-all-tissue-signal-removed-small.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset gtex "
        "--correction split_signal "
        "--range_size .01 "

rule gtex_binary_prediction:
    threads: 4
    input:
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex.{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
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
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex-signal-removed.{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/gtex-signal-removed.{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset gtex "
        "--correction split_signal "

rule gtex_binary_prediction_small:
    threads: 4
    input:
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex-small.{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/gtex-small.{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset gtex "
        "--range_size .01 "

rule gtex_binary_prediction_signal_removed_small:
    threads: 4
    input:
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/gtex-signal-removed-small.{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/gtex-signal-removed-small.{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset gtex "
        "--correction split_signal "
        "--range_size .01 "

rule simulate_data:
    threads: 8
    output:
        "data/batch_sim_data.tsv"
    shell:
        "python src/simulate_be_data.py {output}"

rule simulate_linear_data:
    threads: 8
    output:
        "data/linear_batch_sim_data.tsv"
    shell:
        "python src/simulate_be_data.py {output} --n_nonlinear 0"

rule simulate_no_signal_data:
    threads: 8
    output:
        "data/no_signal_batch_sim_data.tsv"
    shell:
        "python src/simulate_be_data.py {output} --n_linear 0 --n_nonlinear 0 --n_random 2500"

rule sim_prediction:
    threads: 8
    input:
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/sim_dataset.yml",
        data="data/batch_sim_data.tsv"
    output:
        "results/sim-data.{supervised}_{seed}.tsv"
    shell:
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
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
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/sim-data-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--correction split_signal "
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
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
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
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/linear-sim-data-signal-removed.{wildcards.supervised}_{wildcards.seed}.tsv "
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
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
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
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
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
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
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
        "python src/predict_tissue.py {input.dataset_config} {input.supervised_model} "
        "results/sim-data-split-signal.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--disable_optuna "
        "--dataset sim "
        "--correction split_signal "

# Reviewer requested rules:
rule subset_recount_data:
    threads: 1
    input:
        "data/no_scrna_counts.tsv"
    output:
        "data/recount_gtex_genes.tsv"
    shell:
        "python src/normalize_recount_gtex_genes.py data/no_scrna_counts.tsv data/gene_lengths.tsv "
        "data/recount_gtex_genes.tsv data/recount_metadata.tsv data/gtex_normalized.tsv"

rule change_gene_order:
    threads: 1
    input:
        "data/recount_gtex_genes.tsv",
        "data/gtex_normalized.tsv"
    output:
        "data/reformatted_recount.tsv"
    shell:
        "python src/reorder_genes.py data/recount_gtex_genes.tsv "
        "data/gtex_normalized.tsv data/reformatted_recount.tsv "

rule pickle_recount_gtex:
    input:
        "data/reformatted_recount.tsv"
    output:
        "data/reformatted_recount.pkl"
    shell:
        "python src/pickle_tsv.py {input} data/reformatted_recount.pkl"

rule recount_to_gtex_binary:
    threads: 4
    input:
        "data/reformatted_recount.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_gtex_genes.yml",
        transfer_config = "dataset_configs/gtex_dataset.yml"
    output:
        "results/recount_transfer.{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    wildcard_constraints:
        tissue1='[a-zA-Z]+_?[a-zA-Z]*',
        tissue2='[a-zA-Z]+_?[a-zA-Z]*'
    shell:
        "python src/transfer_model.py {input.dataset_config} {input.transfer_config} {input.supervised_model} "
        "results/recount_transfer.{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--dataset recount "
        "--disable_optuna "

rule gtex_to_recount_binary:
    threads: 4
    input:
        "data/reformatted_recount.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
        transfer_config = "dataset_configs/recount_gtex_genes.yml",
    output:
        "results/gtex_transfer.{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    wildcard_constraints:
        tissue1='[a-zA-Z]+_?[a-zA-Z]*',
        tissue2='[a-zA-Z]+_?[a-zA-Z]*'
    shell:
        "python src/transfer_model.py {input.dataset_config} {input.transfer_config} {input.supervised_model} "
        "results/gtex_transfer.{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--dataset gtex "
        "--disable_optuna "

rule recount_to_gtex_binary_signal_removed:
    threads: 4
    input:
        "data/reformatted_recount.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_gtex_genes.yml",
        transfer_config = "dataset_configs/gtex_dataset.yml"
    output:
        "results/recount_transfer.split_signal.{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    wildcard_constraints:
        tissue1='[a-zA-Z]+_?[a-zA-Z]*',
        tissue2='[a-zA-Z]+_?[a-zA-Z]*'
    shell:
        "python src/transfer_model.py {input.dataset_config} {input.transfer_config} {input.supervised_model} "
        "results/recount_transfer.split_signal.{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--dataset recount "
        "--disable_optuna "
        "--correction split_signal "

rule gtex_to_recount_binary_signal_removed:
    threads: 4
    input:
        "data/reformatted_recount.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
        transfer_config = "dataset_configs/recount_gtex_genes.yml",
    output:
        "results/gtex_transfer.split_signal.{tissue1}.{tissue2}.{supervised}_{seed}.tsv"
    wildcard_constraints:
        tissue1='[a-zA-Z]+_?[a-zA-Z]*',
        tissue2='[a-zA-Z]+_?[a-zA-Z]*'
    shell:
        "python src/transfer_model.py {input.dataset_config} {input.transfer_config} {input.supervised_model} "
        "results/gtex_transfer.split_signal.{wildcards.tissue1}.{wildcards.tissue2}.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--tissue1 {wildcards.tissue1} "
        "--tissue2 {wildcards.tissue2} "
        "--weighted_loss "
        "--dataset gtex "
        "--disable_optuna "
        "--correction split_signal "

rule recount_transfer_all:
    threads: 8
    input:
        "data/reformatted_recount.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_gtex_genes.yml",
        transfer_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/recount-transfer.all-tissue.{supervised}_{seed}.tsv"
    shell:
        "python src/transfer_model.py {input.dataset_config} {input.transfer_config} {input.supervised_model} "
        "results/recount-transfer.all-tissue.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--dataset recount "
        "--disable_optuna "

rule gtex_transfer_all:
    threads: 8
    input:
        "data/reformatted_recount.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
        transfer_config = "dataset_configs/recount_gtex_genes.yml",
    output:
        "results/gtex-transfer.all-tissue.{supervised}_{seed}.tsv"
    shell:
        "python src/transfer_model.py {input.dataset_config} {input.transfer_config} {input.supervised_model} "
        "results/gtex-transfer.all-tissue.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--dataset gtex "
        "--disable_optuna "

rule recount_transfer_all_signal_removed:
    threads: 8
    input:
        "data/reformatted_recount.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/recount_gtex_genes.yml",
        transfer_config = "dataset_configs/gtex_dataset.yml",
    output:
        "results/recount-transfer.split-signal.all-tissue.{supervised}_{seed}.tsv"
    shell:
        "python src/transfer_model.py {input.dataset_config} {input.transfer_config} {input.supervised_model} "
        "results/recount-transfer.split-signal.all-tissue.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--dataset recount "
        "--disable_optuna "
        "--correction split_signal "

rule gtex_transfer_all_signal_removed:
    threads: 8
    input:
        "data/reformatted_recount.pkl",
        "data/recount_metadata.tsv",
        "data/recount_sample_to_label.pkl",
        "data/gtex_normalized.pkl",
        "data/gtex_sample_attributes.txt",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/gtex_dataset.yml",
        transfer_config = "dataset_configs/recount_gtex_genes.yml",
    output:
        "results/gtex-transfer.split-signal.all-tissue.{supervised}_{seed}.tsv"
    shell:
        "python src/transfer_model.py {input.dataset_config} {input.transfer_config} {input.supervised_model} "
        "results/gtex-transfer.split-signal.all-tissue.{wildcards.supervised}_{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--all_tissue "
        "--weighted_loss "
        "--dataset gtex "
        "--disable_optuna "
        "--correction split_signal "