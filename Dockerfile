FROM allennlp/allennlp:v0.8.1

ENV LIBRARY_PATH /usr/local/lib
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib

COPY ./dist/*.whl /dist/

RUN pip install --upgrade pip \
&& pip install /dist/*.whl
