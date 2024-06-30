FROM continuumio/miniconda3

WORKDIR ./ivp24

COPY . .

RUN apt-get update && apt-get install -y \
    git vim unzip

RUN conda env update --file environment.yml --name base
RUN pip install sphinx-rtd-theme

# Copy relevant data
RUN mkdir ./data ; mkdir ./data/raw
WORKDIR ./data
RUN git clone https://github.com/loressa/Photoacoustic_radiomics_xenografts
RUN cp ./Photoacoustic_radiomics_xenografts/All_* ./
RUN unzip ./Photoacoustic_radiomics_xenografts/ModelsUncorrected.zip -d ./raw
RUN rm -rf ./Photoacoustic_radiomics_xenografts/
WORKDIR ../

RUN pre-commit install

EXPOSE 8888
