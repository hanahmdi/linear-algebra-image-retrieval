import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from image_retrieval import find_template, compress_svd

class TestAlgorithms(unittest.TestCase):
    def test_ssd_location(self):
        rng=np.random.default_rng(11)
        scene=rng.integers(0,256,(80,90),dtype=np.uint8)
        template=scene[25:39,31:44].copy()
        loc,score,_=find_template(scene,template)
        self.assertEqual(loc,(31,25))
        self.assertLess(score,1.0)  # floating-point matching tolerance

    def test_correlation_location(self):
        rng=np.random.default_rng(10)
        scene=rng.integers(0,256,(80,90),dtype=np.uint8)
        template=scene[17:30,41:55].copy()
        loc,score,_=find_template(scene,template,'correlation')
        self.assertEqual(loc,(41,17))
        self.assertGreater(score,0.99)

    def test_svd_monotone_error(self):
        rng=np.random.default_rng(3)
        image=rng.random((32,24))
        _, e1=compress_svd(image,3)
        _, e2=compress_svd(image,12)
        self.assertLessEqual(e2,e1+1e-12)
        _, e_full=compress_svd(image,24)
        self.assertLess(e_full,1e-12)

if __name__=='__main__':unittest.main()
