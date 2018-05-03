FROM allennlp/allennlp:v0.4.0

#ENV LIBRDKAFKA_VERSION 0.11.1

# Install confluent-kafka dependencies
#RUN curl -Lk -o /root/librdkafka-${LIBRDKAFKA_VERSION}.tar.gz https://github.com/edenhill/librdkafka/archive/v${LIBRDKAFKA_VERSION}.tar.gz && \
#    tar -xzf /root/librdkafka-${LIBRDKAFKA_VERSION}.tar.gz -C /root && \
#    cd /root/librdkafka-${LIBRDKAFKA_VERSION} && \
#    ./configure && make && make install && make clean && ./configure --clean

#ENV CPLUS_INCLUDE_PATH /usr/local/include
ENV LIBRARY_PATH /usr/local/lib

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib

COPY . /src
WORKDIR /src

RUN make init spacy install
