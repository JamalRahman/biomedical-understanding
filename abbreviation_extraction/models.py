from typing import Iterable, List, Tuple
import spacy
import numpy as np
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from spacy.matcher import Matcher

spacy.prefer_gpu()

# Rule for Spacy matching to extract words in parentheses
BRACKETED_RULE = [
    {
        "ORTH": "(",
    },
    {"TEXT": {"NOT_IN": [")"]}, "OP": "+"},
    {"ORTH": ")"},
]


def match_abbreviation(short_form: str, long_form: str):
    """Matches an abbreviation to preceeding long form defitions, if one exists.
    Part of the Schwartz & Hearst abbreviation matching algorithm

    Args:
        short_form (str): The abbreviation
        long_formm (str): The preceeding characters in which to search for a match

    Returns:
        str: The original short form's corresponding long form defition.
    """
    sf_index = len(short_form) - 1
    lf_index = len(long_form) - 1

    while sf_index >= 0:
        # Store the next character to match. Ignore case
        curr_char = short_form[sf_index].lower()
        # ignore non alphanumeric characters
        if not curr_char.isalnum():
            sf_index -= 1
            continue

        while (lf_index >= 0 and long_form[lf_index].lower() != curr_char) or (
            (sf_index == 0) and lf_index > 0 and long_form[lf_index - 1].isalnum()
        ):
            lf_index -= 1

        if lf_index < 0:
            return None

        lf_index -= 1
        sf_index -= 1

    return long_form[lf_index + 1 :]


def find_abbreviation_candidates(sentence: spacy.tokens.span.Span, parentheses_words):
    """Finds valid short forms and selects the possible prefix long form text

    Args:
        sentence (spacy.tokens.span.Span): Spacy object containing the text
            and token information
        parentheses_words (spacy.): Matcher output that gives indices of the
            tokens that match the parentheses

    Returns:
        list(Tuple[str,str]): List of Tuples containing the short
        abbreviation form, and the candidates for the definition
    """
    abbreviation_candidates = []

    for _, parentheses_start, parentheses_end in parentheses_words:
        parenthesis_word = sentence[
            parentheses_start + 1 : parentheses_end - 1
        ]  # Excluding the parentheses
        short_len = len(parenthesis_word.text)
        # Criteria given by Schwartz and Hearst to constitute a valid abbreviation
        valid = parenthesis_word.text[0].isalnum()
        valid = any(c.isalpha() for c in parenthesis_word.text) and valid
        valid = (len(parenthesis_word.text.split()) <= 2) and valid
        valid = short_len >= 2 and short_len <= 10 and valid

        if valid:
            # Long form has a max length in the paper, so we start selecting the
            # candidate pool at this number of characters before the brackets
            long_start = max(0, parentheses_start - min(short_len + 5, short_len * 2))
            long_form = sentence[long_start:parentheses_start]

            abbreviation_candidates.append((str(parenthesis_word), str(long_form)))

    return abbreviation_candidates


class SimpleAbbreviationExtractor:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_md")

        # The algorithm treats hyphenated words as single words, hence we do not wish to tokenize them for just this
        # engineering use-case. The handling of data, tokenized or otherwise, can be optimized for downstream tasks in this manner.
        # https://stackoverflow.com/questions/55241927/spacy-intra-word-hyphens-how-to-treat-them-one-word
        inf = list(self.nlp.Defaults.infixes)
        inf = [
            x for x in inf if "-|–|—|--|---|——|~" not in x
        ]  # remove the hyphen-between-letters pattern from infix patterns
        infix_re = compile_infix_regex(tuple(inf))

        def custom_tokenizer(nlp):
            return Tokenizer(
                nlp.vocab,
                prefix_search=nlp.tokenizer.prefix_search,
                suffix_search=nlp.tokenizer.suffix_search,
                infix_finditer=infix_re.finditer,
                token_match=nlp.tokenizer.token_match,
                rules=nlp.Defaults.tokenizer_exceptions,
            )

        self.nlp.tokenizer = custom_tokenizer(self.nlp)
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add("bracketed", [BRACKETED_RULE])

    def find_abbreviations(self, text: str):
        """Implements the rule-based abbreviation
            extraction algorithm proposed by Schwartz and Hearst (2003).

            When given a body of text, this function returns a list of
            source sentences, abbreviations, and their definitions.
        Args:
            text (str): string in which to find the abbreviations

        Returns:
            np.ndarray: An array of elements where each element
                gives an abbreviation along with its in-place sentence
                and its definition.
        """
        processed_text = self.nlp(text)
        all_abbreviations = []

        for sent in processed_text.sents:
            parentheses_words = self.matcher(sent)
            parentheses_with_candidates = find_abbreviation_candidates(
                sent, parentheses_words
            )

            for possible_solution in parentheses_with_candidates:
                abbreviation_definition = match_abbreviation(
                    possible_solution[0], possible_solution[1]
                )
                if abbreviation_definition is not None:
                    all_abbreviations.append(
                        np.array(
                            [sent.text, possible_solution[0], abbreviation_definition]
                        )
                    )

        return np.array(all_abbreviations)


def heuristic_abbreviation_match(found_entities: List[spacy.tokens.span.Span]):
    """
    Simple heuristc that matches short forms with a long form that appears
    previously in the string provided there is no other short form in between

    i.e - matches adjacent short/long forms

    Args:
        found_entities (List[spacy.tokens.span.Span}): Ordered list of short/long forms in
        order that they appear in text

    Returns:
        List: List of paired short/long forms
    """
    matches = []

    for i in range(len(found_entities) - 1, 0, -1):
        if found_entities[i].label_ == "SF" and found_entities[i - 1].label_ == "LF":
            matches.append([found_entities[i], found_entities[i - 1]])
    return matches


class BertAbbreviationExtractor:
    def __init__(self, base_name) -> None:
        self.nlp = spacy.load(base_name)

    def find_abbreviations(self, text: str):
        """
            When given a body of text, this function returns a list of
            source sentences, abbreviations, and their definitions.
        Args:
            text (str): string in which to find the abbreviations

        Returns:
            np.ndarray: An array of elements where each element
                gives an abbreviation along with its in-place sentence
                and its definition.
        """
        processed_text = self.nlp(text)
        all_abbreviations = []

        for sent in processed_text.sents:
            found_abbreviations = heuristic_abbreviation_match(sent.ents)

            if found_abbreviations:
                for found_pair in found_abbreviations:
                    all_abbreviations.append(np.array([sent, *found_pair]))

        return np.array(all_abbreviations)
