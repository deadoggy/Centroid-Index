# Centroid-Index Python Version

## Usage:

```python

from centroid_index import centroid_index

CI = centroid_index(data, label_1, label_2)

```

## Parameters

```python
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
                sklearn.neighbors.dist_metrics(except 'pyfunc').
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
```

