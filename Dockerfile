FROM continuumio/miniconda3:latest

RUN useradd -ms /bin/bash puzle
WORKDIR /home/puzle

COPY environment.yml environment.yml
RUN conda env create -f environment.yml
RUN conda init bash
RUN echo "source activate puzle" >> /home/puzle/.bashrc
ENV PATH /opt/conda/envs/puzle/bin:$PATH

COPY puzle puzle
COPY migrations migrations
COPY puzleapp.py config.py boot.sh ./
COPY bin/process_stars.py process_stars.py
RUN chmod 775 process_stars.py
RUN mkdir /home/puzle/logs &&\
    chown -R puzle:puzle /home/puzle/logs &&\
    chmod -R 775 /home/puzle/logs &&\
    mkdir /home/puzle/static &&\
    chown -R puzle:puzle /home/puzle/static &&\
    chmod -R 775 /home/puzle/static &&\
    mkdir /home/puzle/data &&\
    chown -R puzle:puzle /home/puzle/data &&\
    chmod -R 775 /home/puzle/data &&\
    mkdir /home/puzle/data/DR4 &&\
    chown -R puzle:puzle /home/puzle/data/DR4 &&\
    chmod -R 775 /home/puzle/data/DR4 &&\
    mkdir /home/puzle/data/PS1_PSC &&\
    chown -R puzle:puzle /home/puzle/data/PS1_PSC &&\
    chmod -R 775 /home/puzle/data/PS1_PSC &&\
    mkdir -p /home/puzle/astropy_cache/astropy &&\
    chown -R puzle:puzle /home/puzle/astropy_cache &&\
    chmod -R 777 /home/puzle/astropy_cache &&\
    mkdir -p /home/puzle/astropy_config/astropy &&\
    chown -R puzle:puzle /home/puzle/astropy_config &&\
    chmod -R 777 /home/puzle/astropy_config &&\
    chown -R puzle:puzle /home/puzle/astropy_config/astropy &&\
    chmod -R 777 /home/puzle/astropy_config/astropy &&\
    chmod +x boot.sh &&\
    chown -R puzle:puzle ./

ENV XDG_CACHE_HOME /home/puzle/astropy_cache
ENV XDG_CONFIG_HOME /home/puzle/astropy_config

RUN cd /home && git clone https://github.com/MichaelMedford/zort.git && cd /home/zort && git checkout 38a31c74e
ENV PYTHONPATH /home/zort:$PYTHONPATH

COPY data/eta_thresholds.dct /home/puzle/data/eta_thresholds.dct

ENV MPLCONFIGDIR /tmp/
ENV FLASK_APP puzleapp.py
USER puzle
EXPOSE 5000