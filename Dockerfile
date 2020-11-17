FROM continuumio/miniconda3:latest

RUN useradd -ms /bin/bash puzle
WORKDIR /home/puzle

COPY environment.yml environment.yml
RUN conda env create -f environment.yml
RUN conda init bash
RUN echo "source activate puzle" >> /home/puzle/.bashrc
ENV PATH /opt/conda/envs/puzle/bin:$PATH

COPY app app
COPY migrations migrations
COPY puzleapp.py config.py boot.sh ./
RUN mkdir /home/puzle/logs &&\
    chown -R puzle:puzle /home/puzle/logs &&\
    chmod -R 775 /home/puzle/logs &&\
    mkdir /home/puzle/data &&\
    chown -R puzle:puzle /home/puzle/data &&\
    chmod -R 775 /home/puzle/data &&\
    mkdir -p /home/puzle/astropy_cache/astropy &&\
    chown -R puzle:puzle /home/puzle/astropy_cache &&\
    chmod -R 777 /home/puzle/astropy_cache &&\
    mkdir -p /home/puzle/astropy_config/astropy &&\
    chown -R puzle:puzle /home/puzle/astropy_config &&\
    chmod -R 777 /home/puzle/astropy_config &&\
    chmod +x boot.sh &&\
    chown -R puzle:puzle ./

ENV XDG_CACHE_HOME /home/puzle/astropy_cache
ENV XDG_CONFIG_HOME /home/puzle/astropy_config

ENV MPLCONFIGDIR /tmp/
ENV FLASK_APP puzleapp.py
USER puzle
EXPOSE 5000
ENTRYPOINT ["./boot.sh"]