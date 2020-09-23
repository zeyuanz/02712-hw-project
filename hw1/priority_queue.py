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
