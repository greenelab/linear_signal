# This script downloads the compendium data from refine.bio and the manual labels from
# the whistl repository on github

python download_data.py

SCRIPT_DIR=`dirname $0`
DATA_DIR=$SCRIPT_DIR/../data

wget https://github.com/greenelab/whistl/raw/master/data/sample_classifications.pkl.gz -P $DATA_DIR
gunzip $DATA_DIR/sample_classifications.pkl.gz
