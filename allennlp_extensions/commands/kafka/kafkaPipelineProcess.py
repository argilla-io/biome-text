import logging
import sys
import json

from multiprocessing import Process, Event
from typing import Iterable, Dict

logger = logging.getLogger(__name__)

try:
    import confluent_kafka as kafka
except Exception:
    logger.warning("confluent_kafka not found")


class KafkaPipelineProcess(Process):
    def __init__(self, consumer_group: str,
                 consume_from: Iterable[str],
                 publish_to: Iterable[str],
                 boostrap_servers: str):

        super().__init__()

        self._exit = Event()

        self.group = consumer_group
        self.__boostrap_servers = boostrap_servers
        self._topics_from = consume_from
        self._topics_to = publish_to

    @classmethod
    def from_params(cls, params: Dict):
        return cls(**params)

    def set_transformation(self, transform):
        self._action = transform

    def stop(self):
        self._exit.set()
        logger.info('shutting down...')

    def is_running(self):
        return not self._exit.is_set()

    def run(self):
        self._exit = Event()
        try:
            consumer = self._create_consumer()
            producer = self._create_producer()
            topics = self.get_topics()

            consumer.subscribe(topics)
            logger.info('Subscrite to topics %s' % topics)

            while self.is_running():
                msg = consumer.poll()

                if not msg.error():
                    self.handle_message(msg, producer)

                elif msg.error().code() != kafka.KafkaError._PARTITION_EOF:
                    logger.error(msg.error())
                    self.stop()

            self._close_consumer(consumer)
            self._close_producer(producer)
        except:
            logger.warning('Error running pipeline %s' % sys.exc_info())

    def handle_message(self, msg, producer):
        # TODO configure serializers & deserializers
        global output
        try:
            output = self._action(json.loads(msg.value().decode('utf-8')))
        except Exception as e:
            # TODO manage error
            logger.error(e)
            return

        for topic in self._topics_to:
            try:
                producer.produce(topic, json.dumps(output))
            except Exception as e:
                logger.error(e)

    def _create_consumer(self):
        return kafka.Consumer({
            'bootstrap.servers': self.__boostrap_servers,
            'group.id': self.group,
            'default.topic.config': {'auto.offset.reset': 'smallest'}
        })

    def _close_producer(self, producer: kafka.Producer):
        producer.flush()
        logger.info('producer closed')

    def _close_consumer(self, consumer: kafka.Consumer):
        consumer.close()
        logger.info('consumer closed')

    def get_topics(self) -> Iterable[str]:
        return self._topics_from

    def _create_producer(self) -> kafka.Producer:
        return kafka.Producer({'bootstrap.servers': self.__boostrap_servers})
