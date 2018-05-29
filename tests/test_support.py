import unittest

from distributed import LocalCluster


class DaskSupportTest(unittest.TestCase):
    cluster: LocalCluster = None

    @classmethod
    def setUpClass(cls):
        if not DaskSupportTest.cluster:
            DaskSupportTest.cluster = LocalCluster(scheduler_port=8786)

    @classmethod
    def tearDownClass(cls):
        DaskSupportTest.cluster.close()
        DaskSupportTest.cluster = None
