import spacy
import pandas as pd
import random
import warnings

from tqdm import tqdm
from spacy.util import minibatch, compounding
from spacy.training import Example

from thinc.api import Adam
from sklearn.model_selection import train_test_split
from utils import process_PLOD

from config import TESTFLAG, TRAINED_MODEL_OUTPUT_PATH, PLOD_LOCATION, NUM_EPOCHS

warnings.filterwarnings("ignore")
spacy.prefer_gpu()

# A fairly textbook Adam optimizer. We're not here to optimize the performance today
optimizer = Adam(
    learn_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-08,
    L2=1e-4,
    grad_clip=1.0,
    use_averages=True,
    L2_is_weight_decay=True,
)

if __name__ == "__main__":

    formatted_data = process_PLOD(PLOD_LOCATION)
    nlp = spacy.load("en_core_web_trf", exclude=["ner"])
    ner = nlp.create_pipe("ner")
    nlp.add_pipe("ner", last=True)

    # Really quick dirty hack for the purposes of local testing, only doing this because
    # its a one-off case_study
    if TESTFLAG:
        formatted_data = formatted_data[:10000]

    # Training the model requires the data to be preprocessed and wrapped into an Example object
    # Iterate through the well-formatted data, and wrap it into a model-ready Example object
    examples = []
    for text, annots in tqdm(formatted_data):
        try:
            examples.append(Example.from_dict(nlp.make_doc(text), annots))
        except ValueError as e:
            continue

    ner.add_label("SF")
    ner.add_label("LF")

    train_data, val_data = train_test_split(examples, test_size=0.3)
    dev_data, test_data = train_test_split(val_data, test_size=0.5)

    # Number of traininig epochs
    n_iter = NUM_EPOCHS

    # We want to finetune the transformer's word embeddings, its NER tagging, and BERT's subword token splitting (wordpiecing)
    select_pipes = ["ner", "trf_wordpiecer", "trf_tok2vec"]

    # Spacy's canonical training loop
    # Continue only with the selected pipelines:
    with nlp.select_pipes(enable=select_pipes):

        nlp.initialize(lambda: train_data)

        for i in range(n_iter):
            random.shuffle(train_data)

            losses = {}  # for logging

            # Again, we're not here to maximize performance, this is just a vague guess at batch sizes
            # Cross-validation & hyperparameter tuning would be an extension goal
            for batch in tqdm(
                minibatch(train_data, spacy.util.compounding(32, 128, 1.002))
            ):

                nlp.update(batch, losses=losses, drop=0.5, sgd=optimizer)

            # Optionally could now specifically run the model on the dev set with no optimizer
            # to determine validation *loss* during training
            # This can be useful for examining the performance & training behaviour of the model

            # For now, inspect the increasing performance (prec, rec, f1) on the dev set

            with nlp.select_pipes(enable="ner"):
                scores = nlp.evaluate(dev_data)

            print("Epoch", i)
            print("Training Loss:", losses)
            print("Validation Scores:", scores)
            print("----------------------------------")

    with nlp.select_pipes(enable="ner"):
        test_scores = nlp.evaluate(test_data)

    print("Test Scores:", test_scores)

    nlp.to_disk(TRAINED_MODEL_OUTPUT_PATH)
