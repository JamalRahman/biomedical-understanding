from typing import Dict
import lxml.etree as et
import pandas as pd
import ast

from config import TARGET_ELEMENTS


def read_articles(read_location: str) -> et._Element:
    """Reads xml file from disk and returns the root node

    Args:
        read_location (str): Relative path to the file

    Returns:
        et._Element: XML root element
    """
    xml_tree = et.parse(read_location)
    return xml_tree.getroot()


def load_articles(
    read_location: str, target_elements: Dict[str, str] = TARGET_ELEMENTS
) -> pd.DataFrame:
    """Loads all articles from an XML file and returns a dataframe
    with their IDs, title, and abstract information

    Args:
        read_location (str): The file location to read articles from

        target_elements (dict(str, str)): A dictionary of XML element paths to extract
            for each article. The keys are the XML element paths, the values are the
            names for the dataframe's columns corresponding to each XML element.
    Returns:
        pandas.DataFrame: The dataframe containing all articles from the file
    """

    xml_root = read_articles(read_location)
    extracted_article_data = []
    for article in xml_root:
        article_data_as_text = []

        # Find the desired nodes and for each one convert their data to simple text
        for element in map(article.find, target_elements.keys()):
            if element is not None:  # xml tostring functionality is not nullsafe
                element = et.tostring(
                    element, encoding="unicode", method="text", with_tail=False
                )
            article_data_as_text.append(element)

        extracted_article_data.append(article_data_as_text)

    df = pd.DataFrame(extracted_article_data, columns=target_elements.values())

    return df


def clean_data(articles: pd.DataFrame) -> pd.DataFrame:
    """Removes non-NLP related errors from the articles data
        In this narrow use-case, the only errors are rows with missing abstracts,
        which are removed

    Args:
        articles (pd.DataFrame): DataFrame containing article information

    Returns:
        pd.DataFrame: DataFrame containing processed article information
    """
    return articles.dropna(subset=["abstract"])


def is_evaluatable(s):

    try:
        ast.literal_eval(s)
        return True
    except ValueError:
        return False


def process_PLOD(path):
    """A messy data processing function
    compiled from the experimentation notebook
    Would be tidied for prod

    Processed PLOD data into a Spacy form

    Args:
        path (str): file location of PLOD dataset TSV
    Returns:
        List: List of sentences with their labelled entities as per Spacy's format
    """
    df = pd.read_csv(path, encoding="utf-8", sep="\t")
    # Lowercase columns for easier manipulation
    df.columns = df.columns.str.lower()
    df = df[
        df["abbreviation_indexes"].apply(is_evaluatable)
        & df["long-form_indexes"].apply(is_evaluatable)
    ]

    df["abbreviation_indexes"] = [
        ast.literal_eval(each) for each in list(df["abbreviation_indexes"])
    ]
    df["long-form_indexes"] = [
        ast.literal_eval(each) for each in list(df["long-form_indexes"])
    ]

    reformatted_abbrevs = []
    for doc_abbrevs in df["abbreviation_indexes"]:
        temp = [(*indices, "SF") for indices in doc_abbrevs]
        reformatted_abbrevs.append(temp)

    reformatted_longs = []
    for doc_abbrevs in df["long-form_indexes"]:
        temp = [(*indices, "LF") for indices in doc_abbrevs]
        reformatted_longs.append(temp)

    df["sf"] = reformatted_abbrevs
    df["lf"] = reformatted_longs

    entities = [sf + lf for sf, lf in zip(reformatted_abbrevs, reformatted_longs)]

    formatted_data = []
    for i, doc in enumerate(df["segment"]):
        curr = (doc, {"entities": entities[i]})
        formatted_data.append(curr)

    return formatted_data
