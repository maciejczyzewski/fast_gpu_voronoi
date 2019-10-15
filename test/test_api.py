from fast_gpu_voronoi import Instance
from fast_gpu_voronoi.jfa import JFA, JFA_mod, JFA_star

arr = [JFA, JFA_mod, JFA_star]

I = Instance(alg=arr[2], x=1000, y=1000, \
        pts=[[ 7,14], [33,34], [27,10],
             [35,10], [23,42], [34,39]])
print(I.run())

def test_shape():
    print(I.M.shape)
    assert I.M.shape == (50, 50, 1)

def test_result():
    assert list(I.M[23][20:35]) == [
        [3], [3], [3], [3],
        [2], [2], [2], [2], [2], [2], [2], [2],
        [5], [5], [5]]
