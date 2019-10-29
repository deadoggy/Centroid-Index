import numpy as np
from sklearn.neighbors import DistanceMetric



def _label_to_list(data, label):
    '''
        generate list for each cluster

        argv:
            @data: 
                list, original data
            @label:
                1-D list, size=len(data), label of data points

        return:
            2-D list 
    '''

    list_clusters = [[] for i in range(len(data))]
    max_label = -1
    min_label = np.inf

    for i in range(len(data)):
        list_clusters[label[i]].append(data[i])
        max_label = label[i] if label[i]>max_label else max_label
        min_label = label[i] if label[i]<min_label else min_label

    return list_clusters[min_label : max_label+1]

def _center_as_prototype(cluster):
    '''
        calculate the center of  cluster as its prototype

        argv:
            @cluster:
                2-D list, each element is a data point
        return:
            1-D list, center of cluster
    '''

    np_cluster = np.array(cluster)
    return (np_cluster.sum(axis=0)/np_cluster.shape[0]).tolist()

def _sum_orphan(clusters_1, clusters_2, metric, prototype):
    '''
        calcualte sum of orphan values from clusters_1 to clusters_2

        argv:
            @clusters_1:
                3-D list, each of element is a list of all data points in a
                cluster
            @clusters_2:
                3-D list, same with clusters_1
            @metric:
                string or function, same with @metric in centroid_index
            @prototype:
                string or function, same with @prototype in centroid_index

        return:
            sum of orphan values from clusters_1 to clusters_2
    ''' 
    
    # calculate prototype of each cluster
    proto_func = _center_as_prototype if prototype=='center' else prototype
    c1_proto = np.array([proto_func(cls) for cls in clusters_1])
    c2_proto = np.array([proto_func(cls) for cls in clusters_2])

    # generate distance calculating function
    if type(metric)==str:
        dm = DistanceMetric.get_metric(metric)
        dist = lambda x,y : dm.pairwise(np.array([x]), np.array([y]))[0][0]
    else:
        dist = metric
    
    # calculate a matrix with shape (len(c1_proto), len(c2_proto)),
    # element (x,y) is 1 if c2_proto[y] is the nearest prototype of c1_proto[x]
    nearest_mat = np.zeros((c1_proto.shape[0], c2_proto.shape[0]))
    for i in range(c1_proto.shape[0]):
        nearest_idx = -1
        nearest_dist = np.inf
        for j in range(c2_proto.shape[0]):
            dist_val = dist(c1_proto[i], c2_proto[j]) 
            if dist_val < nearest_dist:
                nearest_dist = dist_val
                nearest_idx = j
        nearest_mat[i][nearest_idx] = 1

    # return how many orphan in c2_proto
    sum_mat = nearest_mat.sum(axis=0)
    return sum_mat.shape[0] - np.count_nonzero(sum_mat) 

def centroid_index(data, label_1, label_2, metric='euclidean', prototype='center', symmetric=True):
    '''
        calculate centroid index of two clustering results of centroid index

        argv:
            @data: 
                list, original data
            @label_1: 
                1-D list, size=len(data), label of first clustering result
            @label_2:
                1-D list, size=len(data), label of second clustering result
            @metric:
                string or function, metric to calculate distance between
                prototypes. For string please follow the rules of 
                sklearn.neighbors.dist_metrics(expect 'pyfunc').
                If the metric is user defined, please assign this arg as a
                function which takes two prototypes as input. The prototypes 
                are given by @prototype
            @prototype:
                string or function, method to calculate prototype of a cluster. 
                Default is 'center' which means use the sum(all vectors)/cluster_size 
                as the prototype.  
                Otherwise, assign a function which take a list and
                ouput a prototype. If @metric is string, prototypes should
                be 1-D list; If @metric is function, prototypes should be
                valid input of @metric
            @symmetric:
                boolean, if True return max(CI(label_1, label_2), CI(label_2,
                label_1)); else return CI(label_1, label_2)
        return:
            int, CI values
    '''

    # generate clusters' list
    clusters_1 = _label_to_list(data, label_1)
    clusters_2 = _label_to_list(data, label_2)     
    
    # calculate sum of orphan values 
    if symmetric:
        return max(_sum_orphan(clusters_1, clusters_2, metric, prototype), \
                   _sum_orphan(clusters_2, clusters_1, metric, prototype))
    else:
        return _sum_orphan(clusters_1, clusters_2, metric, prototype)


