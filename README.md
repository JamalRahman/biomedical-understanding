# Biomedical Document Understanding - Jamal Rahman

In this first-pass use case, we wish to automatically detect abbreviations. We apply a rules-based and a ML-based approach to automatically detecting abbreviations. We implement the Schwartz and Hearst (2003) rules-based algorithm, and we apply transfer learning to RoBERTa (Liu et al, 2019) for a more powerful approach to abbreviation detection.

The rules-based approach, and the inference of the ML model, are executed in the `abbreviation_extraction.main` module.

The ML model training is given in the `spacy-ner-pipeline.ipynb` notebook, and the training pipeline can also be executed by the `abbreviation_extraction.train` module.

The project is Dockerized, using a CUDA-enabled pytorch base image. If CUDA (11.6) is not available, the image will default to working on CPU only

The output TSV files are given in the `data/out` folder. See the 'Project Structure' section for more details on the directory structure

The src code uses [Black](https://github.com/psf/black) for code style. I started flake8 linting and mypy type testing but did not exhaustively type-check/etc as this is not going into production.

Configuration is done by setting/passing env vars as per the [Twelve 12 Factor App](https://12factor.net/) design.


---

## Project Structure

The project directory structure is given below:

This project uses a subset of the  [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project structure for ease of use and good organisation practices. The majority of the Cookiecutter project structure would be overkill for this simple case study, so I've trimmed it down to the following:

    ├── Dockerfile                                     
    ├── README.md
    |
    ├── abbreviation_extraction                     # Source code package
    │   ├── config.py                               # For reading env variables
    │   ├── data.py                                 # Processing 3rd party datasets
    │   │
    │   ├── main.py                                 # Program for running abbreviation extraction
    │   │
    │   ├── models.py                               # NLP logic
    │   ├── train.py                                # Notebook training implemented in an executable module
    │   └── utils.py                                # Utils!
    ├── data
    │   ├── out
    │   │   ├── ml_based_abbreviations.tsv          # ML-based output file
    │   │   └── rule_based_abbreviations.tsv        # Rule-based output file
    │   ├── processed                               # Interim processed data
    │   └── raw                                     # Immutable data dump
    |
    ├── models                                      # Serialized models
    │   └── binaries
    │      └── roberta-abbrev-identifier.tar.gz    # Compressed model binary
    |
    ├── notebooks
    │   ├── pytorch-bert-finetune.ipynb             # Experiments with huggingface/pytorch
    │   └── spacy-ner-pipeline.ipynb                # Full ML training notebook
    |
    ├── references                                  # Reference material
    |
    └── requirements.txt                            # Dependencies

---

## Running the Project

This project runs through Docker containers.

Pre-requisites:

* Docker
* [Git LFS](https://git-lfs.github.com/) -aware system (needed for the large trained model)
* *Optional: CUDA >=11.6 -enabled host system, with appropriate Nvidia CUDA drivers for the executable environment/machine.*

Start at the root directory of the project (The level that contains this README). You can build an image of the project using the following command:


```
    docker build -t biomedical-understanding .
```

The image is configured to use CUDA **11.6**, if it is available on the host machine. If it is available, add the `--gpus all` flag to the docker run command. If you are presented with a CUDA-based error, remove the flag and run the container without attempting GPU support.

Run the Docker image as a container using the following command:

```
    docker run --rm -it [--gpus all] -t biomedical-understanding:latest /bin/bash
```

Rather than launching directly into a python app, we launch into the shell - this works as a dev environment but also so we can launch into a few different pielines easily.


From within the container, you can run main.py to use the algorithms/models to process articles.xml and output the tsv files. I already gave my own runs' tsv files but you can generate new ones:

```
    user@<containerid>:/app $ python3 abbreviation_extraction/main.py
```

New outputs are given the suffix `new`. I.e - `new_rule_based_abbreviations.tsv`.

Use the [Docker cp command](https://docs.docker.com/engine/reference/commandline/cp/) if you want to copy files/folders from the container back to your local host machine.

### Training the ML model

The ML training can be executed via the Jupyter Notebook or the python executable pipeline.

If your system doesn't have CUDA (11.6), you can run the pipeline locally but it won't use GPU acceleration and will be slow!

If your host system has CUDA 11.6 drivers, it'll be able to process this pipeline using your GPU/TPU. To execute the ML training pipeline from within the docker container, run:

```
    user@<containerid>:/app $ python3 abbreviation_extraction/train.py
```

---

## Future points for development

### Part 1

Only implemented \<long form> (\<short form>) as a shortcut - its by far the most common. Todo: Expand this to \<short form> (\<long form>).

### Part 2

The `spacy-ner-pipeline.ipynb` discusses many of the possible avenues for future work in the ML-space. Highlights include:

* Creating a bespoke abbreviation expansion dataset
  * A labelled dataset in the form of our output, where a short form and long form are explicitly paired. Our data merely points out where short forms and long forms are.
  * **We can thereafter score the overall performance of the final abbreviation-expansion system.**
  * Training a supervised candidate selection model
* Create a meta-learner that combines candidate-selection methods
* Create a span-selection method (akin to question answering models)
* Use other architectures
* More data engineering, some of our tokenization doesn't match the PLOD entity labels which means there is no defined behaviour for how to set entity labels for words that cannot be automatically aligned.
  * Implement an alignment method when entity labels don't match tokens

### Engineering

Unit tests, CI with mypy & flake tests for automated testing

Slight refactoring of different abbreviation extraction methods