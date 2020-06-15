from fast_gpu_voronoi import Instance, TestInstance
from fast_gpu_voronoi.jfa import JFA, JFA_mod, JFA_star, Brute, JFA_test

import numpy as np

arr = [JFA, JFA_mod, JFA_star, Brute, JFA_test]

np.random.seed(123)
pts = np.random.randint(0, 1000, ((1<<10), 2))
I = TestInstance(alg=arr[4], x=1000, y=1000, \
        pts=pts)
print(I.run(method="square", step_as_powers=True, step_size=2, noise=True))

def test_shape():
    print(I.M.shape)
    assert I.M.shape == (50, 50, 1)

def test_result():
    assert list(I.M[23][20:35]) == [
        [3], [3], [3], [3],
        [2], [2], [2], [2], [2], [2], [2], [2],
        [5], [5], [5]]
