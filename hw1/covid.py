import sys

import numpy as np


# country class, with index and distance to destination
class country:
    def __init__(self, ind, distance):
        self.ind = ind
        self.dist = distance

# priority queue implementation
class priority_queue:
    def __init__(self):
        self.queue = []

    # swap two elements 
    def swap(self, i, j):
        self.queue[i], self.queue[j] = self.queue[j], self.queue[i]

    # compare two elements 
    def less(self, i, j):
        if self.queue[i].dist < self.queue[j].dist:
            return True
        return False

    # compute len of array
    def len(self):
        return len(self.queue)

    # add country element to array, without fix, dangerous
    def add(self, country):
        self.queue.append(country)

    # push countary to array, with fix
    def push(self, ele):
        self.queue.append(ele)
        up(len(self.queue)-1)

    # pop element (min)
    def pop(self):
        ele = self.queue[0]
        self.swap(0, len(self.queue)-1)
        self.queue = self.queue[:len(self.queue)-1]
        self.down(0)
        return ele

    # fix element with a new key
    def fix(self, j, newDist):
        self.queue[j].dist = newDist
        check = self.down(j)
        if check == False:
            self.up(j)

    # move element up, if possible
    def up(self, j):
        while True:
            i = (j - 1) // 2 
            if j == 0 or i == j or self.less(i ,j):
                break
            self.swap(i, j)
            j = i

    # move element down, if possible
    def down(self, j):
        n = len(self.queue)
        j0 = j
        while True:
            left = j * 2 + 1
            if left >= n:
                break
            right = left + 1
            if right < n and self.less(right, left):
                j = right
            if self.less(left, j) == False:
                break
            self.swap(left, j)
            j = left
        return j > j0 

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
