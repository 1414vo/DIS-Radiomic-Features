FROM continuumio/miniconda3

WORKDIR ./ivp24

COPY . .

RUN apt-get update && apt-get install -y \
    git vim unzip

RUN conda env update --file environment.yml --name base
RUN pip install sphinx-rtd-theme

# Copy relevant data
RUN mkdir ./data ; mkdir ./data/raw; cd ./data
RUN wget https://github.com/loressa/Photoacoustic_radiomics_xenografts
RUN cp ./Photoacoustic_radiomics_xenografts/All_* ./
RUN unzip ./Photoacoustic_radiomics_xenografts/ModelsUncorrected.zip -d ./data/raw
RUN rm -rf ./Photoacoustic_radiomics_xenografts/

RUN pre-commit install

EXPOSE 8888
