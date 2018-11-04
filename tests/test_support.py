import unittest

from distributed import LocalCluster
import dask
import dask.threaded


class DaskSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dask.config.set(scheduler='threads')

    @classmethod
    def tearDownClass(cls):
        pass
