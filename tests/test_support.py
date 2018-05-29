import unittest

from distributed import LocalCluster


class DaskSupportTest(unittest.TestCase):
    cluster: LocalCluster = None

    @classmethod
    def setUpClass(cls):
        if not DaskSupportTest.cluster:
            try:
                DaskSupportTest.cluster = LocalCluster(scheduler_port=8786)
            except:
                ''''''

    @classmethod
    def tearDownClass(cls):
        try:
            DaskSupportTest.cluster.close()
            DaskSupportTest.cluster = None
        except:
            ''''''
