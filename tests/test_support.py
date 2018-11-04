import unittest

from distributed import LocalCluster
import dask
import dask.threaded


class DaskSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dask.set_options(get=dask.threaded.get)

    @classmethod
    def tearDownClass(cls):
        pass
