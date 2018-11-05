FROM allennlp/allennlp:v0.7.1

ENV LIBRARY_PATH /usr/local/lib

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib

COPY . /src
WORKDIR /src

RUN make init spacy install
