LFR benchmark datasets generating step:

1. https://blog.csdn.net/weixin_42254818/article/details/80515908

2.  My datasets are:

benchmark -N 1000 -k 50 -maxk 100 -mu (0.1-1) -minc 100 -maxc 100 

3. Data.py:  Rearrange the multilayer nodes to ensure the nodes in every layer have the same cluster result  (Because the number of nodes in each cluster each layer are all 100)

