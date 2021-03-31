
DATASETS, = glob_wildcards("dataset_configs/{dataset}.yml")
SUPERVISED, = glob_wildcards("model_configs/supervised/{supervised}.yml")
IMPUTE, = glob_wildcards("model_configs/imputation/{impute}.yml")
UNSUPERVISED, = glob_wildcards("model_configs/unsupervised/{unsupervised}.yml")
SEMISUPERVISED, = glob_wildcards("model_configs/semi-supervised/{semisupervised}.yml")
NUM_SEEDS = 5

wildcard_constraints:
    # Random seeds should be numbers
    seed="\d+"

ruleorder:
    # Fix ambiguity with label wildcard
    single_label_unsupervised > single_label

ruleorder: subset_label_unsupervised > subset_label

rule all:
    input:
        # Pickled input dataframe
        "data/subset_compendium.pkl",
        # all_label_comparisons outputs
        expand("results/all_labels.{supervised}.{dataset}.{seed}.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # all_label_comparison_unsupervised
        expand("results/all_labels.{unsupervised}.{supervised}.{dataset}.{seed}.tsv",
               unsupervised=UNSUPERVISED,
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # all_label_comparisons semi-supervised
        expand("results/all_labels.{semisupervised}.{dataset}.{seed}.tsv",
               semisupervised=SEMISUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # single_label
        expand("results/single_label.sepsis.{supervised}.{dataset}.{seed}.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # single_label_unsupervised
        expand("results/single_label.sepsis.{unsupervised}.{supervised}.{dataset}.{seed}.tsv",
               unsupervised=UNSUPERVISED,
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS),
               ),
        # single_label semi-supervised
        expand("results/single_label.sepsis.{semisupervised}.{dataset}.{seed}.tsv",
               semisupervised=SEMISUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # subset_label
        expand("results/subset_label.sepsis.{supervised}.{dataset}.{seed}.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # subset_label_unsupervised
        expand("results/subset_label.sepsis.{unsupervised}.{supervised}.{dataset}.{seed}.tsv",
               unsupervised=UNSUPERVISED,
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS),
               ),
        # subset_label batch effect corrected
        expand("results/subset_label.sepsis.{supervised}.{dataset}.{seed}.be_corrected.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # subset_label semi-supervised
        expand("results/subset_label.sepsis.{semisupervised}.{dataset}.{seed}.tsv",
               semisupervised=SEMISUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # subset_tb
        expand("results/subset_label.tb.{supervised}.{dataset}.{seed}.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # subset_tb_unsupervised
        expand("results/subset_label.tb.{unsupervised}.{supervised}.{dataset}.{seed}.tsv",
               unsupervised=UNSUPERVISED,
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS),
               ),
        # subset_label tb semi-supervised
        expand("results/subset_label.tb.{semisupervised}.{dataset}.{seed}.tsv",
               semisupervised=SEMISUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # subset_label tb batch effect corrected
        expand("results/subset_label.tb.{supervised}.{dataset}.{seed}.be_corrected.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # subset_all sepsis batch effect corrected
        expand("results/subset_all.sepsis.{supervised}.{dataset}.{seed}.be_corrected.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # subset_all tb batch effect corrected
        expand("results/subset_all.tb.{supervised}.{dataset}.{seed}.be_corrected.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # keep_ratios sepsis
        expand("results/keep_ratios.sepsis.{supervised}.{dataset}.{seed}.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # keep_ratios tb
        expand("results/keep_ratios.tb.{supervised}.{dataset}.{seed}.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # keep_ratios lupus
        expand("results/keep_ratios.lupus.{supervised}.{dataset}.{seed}.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # small_subsets sepsis
        expand("results/small_subsets.sepsis.{supervised}.{dataset}.{seed}.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # small_subsets tb
        expand("results/small_subsets.tb.{supervised}.{dataset}.{seed}.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # keep_ratios sepsis be_corrected
        expand("results/keep_ratios.sepsis.{supervised}.{dataset}.{seed}.be_corrected.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # keep_ratios tb be_corrected
        expand("results/keep_ratios.tb.{supervised}.{dataset}.{seed}.be_corrected.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # keep_ratios lupus be_corrected
        expand("results/keep_ratios.lupus.{supervised}.{dataset}.{seed}.be_corrected.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # small_subsets sepsis be_corrected
        expand("results/small_subsets.sepsis.{supervised}.{dataset}.{seed}.be_corrected.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # small_subsets tb be_corrected
        expand("results/small_subsets.tb.{supervised}.{dataset}.{seed}.be_corrected.tsv",
               supervised=SUPERVISED,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # basic imputation
        expand("results/impute.{impute}.{dataset}.{seed}.tsv",
               impute=IMPUTE,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),
        # uncorrected imputation
        expand("results/impute.{impute}.{dataset}.{seed}.uncorrected.tsv",
               impute=IMPUTE,
               dataset=DATASETS,
               seed=range(0,NUM_SEEDS)
               ),


rule pickle_compendium:
    input:
        "data/subset_compendium.tsv"
    output:
        "data/subset_compendium.pkl"
    shell:
        "python saged/pickle_tsv.py {input} {output}"


rule all_label_comparison:
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        # There is a dot instead of an underscore here because I can't think of
        # a good regex way to differentiate between config file and dataset file names
        "results/all_labels.{supervised}.{dataset}.{seed}.tsv"
    shell:
        "python saged/all_label_comparison.py {input.dataset_config} {input.supervised_model} "
        "results/all_labels.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed}"

rule all_label_comparison_unsupervised:
    # As far as I can tell the logic required to force snakemake to do optional
    # flags only part of the time would be messier than just writing an extra rule
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
        unsupervised_model = "model_configs/unsupervised/{unsupervised}.yml",
    output:
        # There is a dot instead of an underscore here because I can't think of
        # a good regex way to differentiate between config file and dataset file names
        "results/all_labels.{unsupervised}.{supervised}.{dataset}.{seed}.tsv"
    shell:
        "python saged/all_label_comparison.py {input.dataset_config} {input.supervised_model} "
        "results/all_labels.{wildcards.unsupervised}.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--unsupervised_config {input.unsupervised_model} "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed}"

rule all_label_comparison_semisupervised:
    input:
        "data/subset_compendium.pkl",
        semi_supervised_model = "model_configs/semi-supervised/{semisupervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        # There is a dot instead of an underscore here because I can't think of
        # a good regex way to differentiate between config file and dataset file names
        "results/all_labels.{semisupervised}.{dataset}.{seed}.tsv"
    shell:
        "python saged/all_label_comparison.py {input.dataset_config} {input.semi_supervised_model} "
        "results/all_labels.{wildcards.semisupervised}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--semi_supervised "

rule single_label:
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/single_label.{label}.{supervised}.{dataset}.{seed}.tsv"
    shell:
        "python saged/single_label_prediction.py {input.dataset_config} {input.supervised_model} "
        "results/single_label.{wildcards.label}.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--label {wildcards.label} "
        "--negative_class healthy "

rule single_label_unsupervised:
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
        unsupervised_model = "model_configs/unsupervised/{unsupervised}.yml",
    output:
        "results/single_label.{label}.{unsupervised}.{supervised}.{dataset}.{seed}.tsv"
    shell:
        "python saged/single_label_prediction.py {input.dataset_config} {input.supervised_model} "
        "results/single_label.{wildcards.label}.{wildcards.unsupervised}.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--unsupervised_config {input.unsupervised_model} "
        "--label {wildcards.label} "
        "--negative_class healthy "

rule single_label_semisupervised:
    input:
        "data/subset_compendium.pkl",
        semi_supervised_model = "model_configs/semi-supervised/{semisupervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/single_label.{label}.{semisupervised}.{dataset}.{seed}.tsv"
    shell:
        "python saged/single_label_prediction.py {input.dataset_config} {input.semi_supervised_model} "
        "results/single_label.{wildcards.label}.{wildcards.semisupervised}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--label {wildcards.label} "
        "--negative_class healthy "
        "--semi_supervised "

rule subset_label:
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/subset_label.{label}.{supervised}.{dataset}.{seed}.tsv"
    shell:
        "python saged/subset_label_prediction.py {input.dataset_config} {input.supervised_model} "
        "results/subset_label.{wildcards.label}.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--label {wildcards.label} "
        "--negative_class healthy "

rule subset_label_unsupervised:
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
        unsupervised_model = "model_configs/unsupervised/{unsupervised}.yml",
    output:
        "results/subset_label.{label}.{unsupervised}.{supervised}.{dataset}.{seed}.tsv"
    shell:
        "python saged/subset_label_prediction.py {input.dataset_config} {input.supervised_model} "
        "results/subset_label.{wildcards.label}.{wildcards.unsupervised}.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--unsupervised_config {input.unsupervised_model} "
        "--label {wildcards.label} "
        "--negative_class healthy "

rule subset_label_semisupervised:
    input:
        "data/subset_compendium.pkl",
        semi_supervised_model = "model_configs/semi-supervised/{semisupervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/subset_label.{label}.{semisupervised}.{dataset}.{seed}.tsv"
    shell:
        "python saged/subset_label_prediction.py {input.dataset_config} {input.semi_supervised_model} "
        "results/subset_label.{wildcards.label}.{wildcards.semisupervised}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--label {wildcards.label} "
        "--negative_class healthy "
        "--semi_supervised"

rule subset_label_batch_effect_correction:
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/subset_label.{label}.{supervised}.{dataset}.{seed}.be_corrected.tsv"
    shell:
        "python saged/subset_label_prediction.py {input.dataset_config} {input.supervised_model} "
        "results/subset_label.{wildcards.label}.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.be_corrected.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--label {wildcards.label} "
        "--negative_class healthy "
        "--batch_correction_method limma"

rule subset_all:
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/subset_all.{label}.{supervised}.{dataset}.{seed}.be_corrected.tsv"
    shell:
        "python saged/subset_all.py {input.dataset_config} {input.supervised_model} "
        "results/subset_all.{wildcards.label}.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.be_corrected.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--label {wildcards.label} "
        "--negative_class healthy "
        "--batch_correction_method limma"

rule keep_ratios:
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/keep_ratios.{label}.{supervised}.{dataset}.{seed}.tsv"
    shell:
        "python saged/keep_ratios.py {input.dataset_config} {input.supervised_model} "
        "results/keep_ratios.{wildcards.label}.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--label {wildcards.label} "
        "--negative_class healthy "

rule keep_ratios_be_correction:
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/keep_ratios.{label}.{supervised}.{dataset}.{seed}.be_corrected.tsv"
    shell:
        "python saged/keep_ratios.py {input.dataset_config} {input.supervised_model} "
        "results/keep_ratios.{wildcards.label}.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.be_corrected.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--label {wildcards.label} "
        "--negative_class healthy "
        "--batch_correction_method limma "

rule small_subsets:
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/small_subsets.{label}.{supervised}.{dataset}.{seed}.tsv"
    shell:
        "python saged/small_subsets.py {input.dataset_config} {input.supervised_model} "
        "results/small_subsets.{wildcards.label}.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--label {wildcards.label} "
        "--negative_class healthy "

rule small_subsets_be_correction:
    input:
        "data/subset_compendium.pkl",
        supervised_model = "model_configs/supervised/{supervised}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/small_subsets.{label}.{supervised}.{dataset}.{seed}.be_corrected.tsv"
    shell:
        "python saged/small_subsets.py {input.dataset_config} {input.supervised_model} "
        "results/small_subsets.{wildcards.label}.{wildcards.supervised}.{wildcards.dataset}.{wildcards.seed}.be_corrected.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--label {wildcards.label} "
        "--negative_class healthy "
        "--batch_correction_method limma "

rule basic_imputation:
    input:
        "data/subset_compendium.pkl",
        imputation_model = "model_configs/imputation/{impute}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/impute.{impute}.{dataset}.{seed}.tsv"
    shell:
        "python saged/impute_expression.py {input.dataset_config} {input.imputation_model} "
        "results/impute.{wildcards.impute}.{wildcards.dataset}.{wildcards.seed}.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
        "--batch_correction_method limma"

rule basic_imputation_uncorrected:
    input:
        "data/subset_compendium.pkl",
        imputation_model = "model_configs/imputation/{impute}.yml",
        dataset_config = "dataset_configs/{dataset}.yml",
    output:
        "results/impute.{impute}.{dataset}.{seed}.uncorrected.tsv"
    shell:
        "python saged/impute_expression.py {input.dataset_config} {input.imputation_model} "
        "results/impute.{wildcards.impute}.{wildcards.dataset}.{wildcards.seed}.uncorrected.tsv "
        "--neptune_config neptune.yml "
        "--seed {wildcards.seed} "
