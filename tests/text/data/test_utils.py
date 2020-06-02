import unittest

from distributed import LocalCluster

from biome.text.data.helpers import (
    configure_dask_cluster,
    close_dask_client,
)


class UtilTests(unittest.TestCase):
    def test_dask_cluster_local(self):

        client = configure_dask_cluster()
        cluster = client.cluster

        self.assertTrue(
            isinstance(cluster, LocalCluster), msg=f"Wrong cluster type: {cluster}"
        )
        self.assertEqual("running", client.status)
        self.assertEqual("running", cluster.status)

        client = configure_dask_cluster(address="localhost:8786")
        self.assertEqual(client, configure_dask_cluster())

        close_dask_client()
        self.assertEqual("closed", client.status)
        self.assertEqual("closed", cluster.status)

    def test_dask_cluster_client(self):
        port = 8788
        cluster = LocalCluster(processes=False, scheduler_port=port)

        client = configure_dask_cluster(address=f"localhost:{port}")
        self.assertEqual(None, client.cluster)
        self.assertEqual("running", client.status)

        close_dask_client()
        self.assertEqual("closed", client.status)
        self.assertEqual("running", cluster.status)

        cluster.close(timeout=10)
        self.assertEqual("closed", cluster.status)
