FROM continuumio/miniconda3

WORKDIR ./ivp24

COPY . .

RUN apt-get update && apt-get install -y \
    git vim

RUN conda env update --file environment.yml --name base
RUN mkdir ./data & mkdir ./data/raw
RUN wget https://github.com/loressa/Photoacoustic_radiomics_xenografts/blob/687874ae17b02f4af035375a189ce408461357c9/ModelsUncorrected.zip
RUN unzip ModelsUncorrected.zip -d ./data/raw & rm ModelsUncorrected.zip
RUN pre-commit install

EXPOSE 8888
