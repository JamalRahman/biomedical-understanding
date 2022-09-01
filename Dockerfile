# syntax=docker/dockerfile:1

FROM cnstark/pytorch:1.12.1-py3.9.12-cuda11.6.2-ubuntu20.04

WORKDIR /app

COPY requirements.txt requirements.txt
ADD models/binaries/roberta-abbrev-identifier.tar.gz models

RUN pip3 install --upgrade pip \
    && pip3 install wheel \
    && pip3 install cupy-cuda116 \
    && pip3 install -r requirements.txt


RUN python3 -m spacy download en_core_web_md \
    && python3 -m spacy download en_core_web_trf

ENV full_iob_data_path=data/processed/new_PLOD_IOB_tagged.conll
ENV simple_output_path=data/out/new_rule_based_abbreviations.tsv
ENV ml_output_path=data/out/new_ml_based_abbreviations.tsv
ENV trained_model_output_path=models/new-roberta-abbrev-identifier

COPY . .
