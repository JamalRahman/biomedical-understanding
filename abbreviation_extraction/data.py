from pathlib import Path
import re

from config import FULL_IOB_DATA_PATH, INDIVIDUAL_IOB_DATA_PATHS


def read_conll(file_path, contains_pos_tags=False):
    file_path = Path(file_path)

    raw_text = file_path.read_text(encoding="utf-8").strip()
    raw_docs = re.split(r"\n\s?\n", raw_text)

    token_docs = []
    tag_docs = []

    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split("\n"):

            if contains_pos_tags:
                token, _, tag = line.split()
            else:
                token, tag = line.split()

            tokens.append(token)
            tags.append(tag)

        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs


def concat_conll_files(out, read_file_paths):
    tokens = []
    tags = []

    for file_path in read_file_paths:
        print(file_path)
        subset_tokens, subset_tags = read_conll(file_path, True)

        tokens += subset_tokens
        tags += subset_tags

    temp_tags = []
    for doc_tags in tags:
        temp_doc_tags = []
        for tag in doc_tags:
            temp_doc_tags.append(tag.replace("AC", "SF").replace("B-O", "O"))
        temp_tags.append(temp_doc_tags)

    tags = temp_tags

    per_doc_pairs = list(zip(tokens, tags))
    per_token_pairs = [list(zip(each[0], each[1])) for each in per_doc_pairs]

    with open(out, "w", encoding="utf-8") as f:
        for doc in per_token_pairs:
            for line in doc:
                linestr = " ".join(map(str, line))
                f.write(f"{linestr}\n")
            f.write("\n")


if __name__ == "__main__":
    concat_conll_files(FULL_IOB_DATA_PATH, INDIVIDUAL_IOB_DATA_PATHS)
