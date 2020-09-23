import sys

import numpy as np

from priority_queue import *


# execute dijkstra algorithm with priority queue implementation
def dijkstra(distMat, s):
    s = s - 1
    dist = [float('Inf')] * len(distMat)
    parent = [None] * len(distMat)
    q = priority_queue()
    for i in range(len(dist)):
        q.add(country(i, dist[i]))
        parent[i] = i
    dist[s] = 0
    q.fix(s, 0)

    while(q.len() != 0):
        curr = q.pop()
        for i in range(q.len()):
            tmp = curr.dist + distMat[q.queue[i].ind ,curr.ind] 
            if tmp < q.queue[i].dist:
                dist[q.queue[i].ind] = tmp
                parent[q.queue[i].ind] = curr.ind
                q.fix(i, tmp)
        
    for i in range(len(dist)):
        dist[i] += distMat[i,i]
    return dist, parent

# compute argmin of a list
def findMin(dist):
    minVal = float('Inf')
    minInd = 0
    for i in range(len(dist)):
        if dist[i] < minVal:
            minVal = dist[i]
            minInd = i
    return minInd

# main function
def main(argv):
    # parse input file into np array
    filename=argv[1] 
    f = open(filename, "r")
    s = int(f.readline())
    distMat = []
    lines = f.readlines()
    distMat = np.zeros((len(lines), len(lines)))
    count = 0
    for line in lines:
        line = line.split(" ")
        for col in range(len(lines)):
            distMat[count, col] = line[col]
            # avoid zero-division error
            if distMat[count, col] == 0.0:
                distMat[count, col] += 1e-10
        count += 1

    # transfer to -log
    distMat = -np.log(distMat)
    # compute distance and parent information from dijstra
    dist, parent = dijkstra(distMat, s)
    # find min distance of dist array
    minInd = findMin(dist)

    # print in requested format
    print(minInd+1,end = '')
    if minInd == s-1:
        return
    print('-',end = '')
    while True:
        if parent[minInd] != s-1:
            print(parent[minInd]+1, end='-')
        else:
            print(parent[minInd]+1, end ='')
            break
        minInd = parent[minInd]

if __name__ == "__main__":
    main(sys.argv)
