FROM continuumio/miniconda3:latest

RUN useradd -ms /bin/bash puzle
WORKDIR /home/puzle

COPY environment.yml environment.yml
RUN conda env create -f environment.yml
RUN conda init bash
RUN echo "conda activate puzle" >> /home/puzle/.bashrc
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
    mkdir /home/puzle/data/ulensdb &&\
    chown -R puzle:puzle /home/puzle/data/ulensdb &&\
    chmod -R 775 /home/puzle/data/ulensdb &&\
    chmod +x boot.sh &&\
    chown -R puzle:puzle ./

ENV XDG_CACHE_HOME /tmp/puzle/astropy_cache
ENV XDG_CONFIG_HOME /tmp/puzle/astropy_config

RUN cd /home &&\
    git clone https://github.com/MichaelMedford/zort.git &&\
    cd /home/zort &&\
    git checkout 38a31c74e &&\
    apt-get install -y musl-dev &&\
    ln -s /usr/lib/x86_64-linux-musl/libc.so /lib/libc.musl-x86_64.so.1
ENV PYTHONPATH /home/zort

COPY data/eta_thresholds.dct /home/puzle/data/eta_thresholds.dct

ENV MPLCONFIGDIR /tmp/
ENV FLASK_APP puzleapp.py
USER puzle
EXPOSE 5000

CMD conda activate puzle