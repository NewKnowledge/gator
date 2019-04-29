
import numpy as np

from gator import predict, init


def test_basic(n_samples=1, n_objects=10):
    
    # force init w/ given num objects returned (init happens on first call otherwise)
    init(n_objects=n_objects)
    
    # should return a list of dicts with one dict per sample and n_objects per dict as {obj_name: score} key, value pairs
    images_array = np.random.randint(0, 255, size=(n_samples, 299, 299, 3))
    objs = predict(images_array)
    assert isinstance(objs, list) and len(objs) == n_samples
    # assert isinstance(objs[0], dict) and len(objs[0]) == n_objects
    assert isinstance(list(objs[0].keys())[0], str)
    assert isinstance(list(objs[0].values())[0], np.float32)
