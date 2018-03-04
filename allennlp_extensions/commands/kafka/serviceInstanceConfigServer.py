import logging

from sanic import Sanic, request, response
from sanic.exceptions import NotFound
from sanic.response import json
from typing import Dict

logger = logging.getLogger(__name__)

try:
    import redis.ConnectionPool as ConnectionPool  # pylint: disable=import-error
    import redis.StrictRedis as StrictRedis  # pylint: disable=import-error
except Exception:
    logger.warning('Redis not found')


class RedisClient():
    def __init__(self, group: str, redis_url: str = 'redis://localhost:6379', db=None):
        self._url = redis_url
        self._group = group
        self._client = self.client(self._url, db)

    def client(self, url, db):
        connection_pool = ConnectionPool.from_url(url, db=db)
        return StrictRedis(connection_pool=connection_pool, retry_on_timeout=True)

    def add_value(self, value):
        if not self._client.sismember(self._group, value):
            self._client.sadd(self._group, value)

    def remove(self, value):
        if not self._client.sismember(self._group, value):
            raise NotFound(value)

        self._client.srem(self._group, value)

    def get_values(self):
        return self._client.smembers(self._group)


class ServiceInstanceConfigServer(object):
    def __init__(self,
                 service_group: str,
                 redis_url: str,
                 port: int = 8000):
        self.app = Sanic(__name__)
        self.port = port
        self.redis_client = RedisClient(service_group, redis_url)

    @classmethod
    def from_params(cls, params: Dict, **kwargs):
        args = {**params, **kwargs}
        return cls(**args)

    def start(self):
        @self.app.route("/subscription", methods=['POST'])
        async def subscribe(req: request.Request) -> response.HTTPResponse:
            data = req.json
            assert data['name']

            self.redis_client.add_value(data['name'])

            return json({
                "message": "ok",
                "subscriptions": self.redis_client.get_values()
            })

        @self.app.route("/subscription", methods=['DELETE'])
        async def unsubscribe(req: request.Request) -> response.HTTPResponse:
            data = req.json
            assert data['name']

            self.redis_client.remove(data['name'])

            return json({
                "message": "ok",
                "subscriptions": self.redis_client.get_values()
            })

        self.app.run(host="0.0.0.0", port=self.port)
