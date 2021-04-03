FROM continuumio/miniconda3:latest

RUN useradd -ms /bin/bash puzle
WORKDIR /home/puzle

COPY environment.yml environment.yml
RUN conda env create -f environment.yml
RUN conda init bash
RUN echo "conda activate puzle" >> /home/puzle/.bashrc
ENV PATH /opt/conda/envs/puzle/bin:$PATH

# install mpi4py

RUN \
    apt-get update        && \
    apt-get install --yes    \
        build-essential      \
        gfortran             \
        wget              && \
    apt-get clean all

ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure                                                             && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

ARG mpi4py=3.0.3
ARG mpi4py_prefix=mpi4py-$mpi4py

RUN \
    wget https://bitbucket.org/mpi4py/mpi4py/downloads/$mpi4py_prefix.tar.gz && \
    tar xvzf $mpi4py_prefix.tar.gz                                           && \
    cd $mpi4py_prefix                                                        && \
    conda run -n puzle python setup.py build                                                   && \
    conda run -n puzle python setup.py install                                                 && \
    cd ..                                                                    && \
    rm -rf $mpi4py_prefix

RUN /sbin/ldconfig

# mpi4py complete

RUN cd /home &&\
    git clone https://github.com/MichaelMedford/zort.git &&\
    cd /home/zort &&\
    git checkout 38a31c74e &&\
    apt-get install -y musl-dev &&\
    ln -s /usr/lib/x86_64-linux-musl/libc.so /lib/libc.musl-x86_64.so.1
ENV PYTHONPATH /home/zort

COPY puzle puzle
COPY migrations migrations
COPY puzleapp.py config.py boot.sh ./
COPY bin/process_stars.py process_stars.py
COPY mpi_slurm_test.py mpi_slurm_test.py
COPY slurm_test.sh slurm_test.sh
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

COPY data/eta_thresholds.dct /home/puzle/data/eta_thresholds.dct

ENV MPLCONFIGDIR /tmp/
ENV FLASK_APP puzleapp.py
USER puzle
EXPOSE 5000

CMD conda activate puzle