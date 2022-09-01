from environs import Env

env = Env()

ARTICLE_READ_PATH = env("article_read_path", "data/raw/articles.xml")

FULL_IOB_DATA_PATH = env("full_iob_data_path", "data/processed/PLOD_IOB_tagged.conll")

INDIVIDUAL_IOB_DATA_PATHS = env(
    "individual_iob_data_paths",
    [
        "data/raw/PLOS-train70-filtered-pos_bio.conll",
        "data/raw/PLOS-val15-filtered-pos_bio.conll",
        "data/raw/PLOS-test15-filtered-pos_bio.conll",
    ],
)

SIMPLE_OUTPUT_PATH = env("simple_output_path", "data/out/rule_based_abbreviations.tsv")
TARGET_ELEMENTS = env(
    "target_elements",
    {
        "MedlineCitation/PMID": "PMID",
        "MedlineCitation/Article/ArticleTitle": "article_title",
        "MedlineCitation/Article/Abstract/AbstractText": "abstract",
    },
)

MODEL_READ_PATH = env("model_read_path", "models/roberta-abbrev-identifier")
ML_OUTPUT_PATH = env("ml_output_path", "data/out/ml_based_abbreviations.tsv")

TESTFLAG = env("testflag", False)

TRAINED_MODEL_OUTPUT_PATH = env(
    "trained_model_output_path", "models/new-roberta-abbrev-identifier"
)
PLOD_LOCATION = env("plod_location", "data/raw/PLOD_filtered.tsv")

NUM_EPOCHS = env("num_epochs", 4)
