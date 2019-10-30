import numpy as np
from centroid_index import _label_to_list
from centroid_index import _sum_orphan
from centroid_index import _center_as_prototype
from centroid_index import centroid_index
from sklearn.cluster import KMeans

def load_test_dataset():
    data = []
    ctr = []
    with open('../dataset/s1.txt') as data_in:
        lines = data_in.readlines()
        for l in lines:
            v = l.strip().split('    ')
            data.append([float(v[0]), float(v[1])])
    with open('../dataset/s1-label.pa') as truth_in:
        truth = [ int(l) for l in  truth_in.readlines() ]
    with open('../dataset/s1-cb.txt') as ctr_in:
        lines = ctr_in.readlines()
        for l in lines:
            c = l.strip().split(' ')
            ctr.append([float(c[0]), float(c[1])])

    return data, truth, ctr

def test_label_to_list():
    data, truth, ctr = load_test_dataset()
    clusters = _label_to_list(data, truth)
    confused_size = [len(cls) for cls in clusters]
    confused_size.sort()
    truth_size = [truth.count(l) for l in set(truth)]
    truth_size.sort()

    assert len(truth_size) == len(confused_size)

    for i in range(len(truth_size)):
        assert truth_size[i] == confused_size[i]

def test_center_as_prototype():
    data, truth, ctr = load_test_dataset()
    ctr.sort(key=lambda x:x[0])

    computed_ctr =  []
    clusters = _label_to_list(data, truth)
    for cls in clusters:
        computed_ctr.append(_center_as_prototype(cls))
    computed_ctr.sort(key=lambda x:x[0])

    for i in range(len(computed_ctr)):
        assert int(computed_ctr[i][0])==ctr[i][0]
        assert int(computed_ctr[i][1])==ctr[i][1]

def test_centroid_index(k):
   data, truth, ctr = load_test_dataset()
   km = KMeans(n_clusters=k).fit(np.array(data))
   label = km.labels_

   #assert 0==centroid_index(data, label, truth)
   print ( centroid_index(data, label, truth) )
   

if __name__ == '__main__':
    test_label_to_list()
    #test_center_as_prototype()
    for i in range(2, 20):
        test_centroid_index(i)


