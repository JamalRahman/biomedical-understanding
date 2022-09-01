from config import MODEL_READ_PATH
from models import BertAbbreviationExtractor
from config import SIMPLE_OUTPUT_PATH, ARTICLE_READ_PATH, ML_OUTPUT_PATH
from models import SimpleAbbreviationExtractor
from utils import clean_data, load_articles
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_abbreviations(extractor, cleaned_articles):
    """Wrapper function that abstracts the logic for:
        - making our input articles' data match with the abbreviation-extractors
        - Iterating across each abstract
        - returning a properly formatted dataframe ready for output.

    Args:
        cleaned_articles (pd.DataFrame): The input articles in a dataframe form

    Returns:
        pd.DataFrame: A DataFrame containing sentence-per-row entries where each row
        matches the task's given schema. Covering every article from the input.
    """

    abstracts_with_ids = cleaned_articles[["PMID", "abstract"]].to_numpy()  # for speed
    abbreviations_with_ids = []

    for keyed_abstract in tqdm(abstracts_with_ids):
        # Get the abbreviations for a single abstract, tracked per sentence
        abbrevs = extractor.find_abbreviations(keyed_abstract[1])

        if abbrevs.size == 0:
            continue
        abbrevs = np.insert(abbrevs, abbrevs.shape[1], keyed_abstract[0], axis=1)
        abbreviations_with_ids.append(abbrevs)

    abbreviations_with_ids = [
        item for sublist in abbreviations_with_ids for item in sublist
    ]

    combined_data = pd.DataFrame(
        abbreviations_with_ids, columns=["sentence", "short_form", "long_form", "PMID"]
    ).merge(cleaned_articles[["PMID", "article_title"]], how="left", on="PMID")[
        ["article_title", "PMID", "sentence", "short_form", "long_form"]
    ]

    return combined_data


if __name__ == "__main__":
    articles = load_articles(ARTICLE_READ_PATH)

    cleaned_articles = clean_data(articles)

    simple_abbreviation_extractor = SimpleAbbreviationExtractor()
    simple_abbreviation_output = get_abbreviations(
        simple_abbreviation_extractor, cleaned_articles
    )
    simple_abbreviation_output.to_csv(SIMPLE_OUTPUT_PATH, sep="\t", index=False)

    bert_abbreviation_extractor = BertAbbreviationExtractor(MODEL_READ_PATH)
    bert_abbreviation_output = get_abbreviations(
        bert_abbreviation_extractor, cleaned_articles
    )
    bert_abbreviation_output.to_csv(ML_OUTPUT_PATH, sep="\t", index=False)
