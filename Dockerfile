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
RUN chmod +x boot.sh

ENV FLASK_APP puzleapp.py

RUN chown -R puzle:puzle ./

USER puzle
EXPOSE 5000
ENTRYPOINT ["./boot.sh"]