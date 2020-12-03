"""
Usage:
    python3 SIR.py input.txt output -v
    -v is optional to print information to logger
"""

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np

# set up log, set level as INFO
logger = logging.getLogger("SIR")
logger.setLevel(level=logging.INFO)

# logger file stream to 'SIR.log'
handler = logging.FileHandler("SIR.log")

# set handler level as INFO
handler.setLevel(logging.INFO)

# set handler log format
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Event:
    def __init__(self, para, val):
        self.para = para
        self.rate = para * val

    def time(self):
        if self.rate < 1e-12:
            return float("inf")
        return np.random.exponential(scale=(1.0 / self.rate))

    def update(self, val):
        self.rate = self.para * val


class State:
    def __init__(self, sus, infected, recover):
        self.s = sus
        self.i = infected
        self.r = recover
        self.N = sus + infected + recover

    def update(self, op):
        if op == "infected":
            self.s -= 1
            self.i += 1
        if op == "recover":
            self.i -= 1
            self.r += 1

    def transit(self, op):
        if op == "out":
            total = self.N
            person = np.random.randint(total) + 1
            if person <= self.s:
                self.s -= 1
                self.N -= 1
                return "s"
            person -= self.s
            if person < self.i:
                self.i -= 1
                self.N -= 1
                return "i"
            self.r -= 1
            self.N -= 1
            return "r"
        else:
            if op == "s":
                self.s += 1
            elif op == "i":
                self.i += 1
            else:
                self.r += 1
            self.N += 1
            return None


def initialization(m, N, k, l1, l2, l3):
    states = [None] * m
    SIR_event = []
    transition_event = []

    for i in range(m):
        states[i] = State(N - k, k, 0)

    for i in range(m):
        tmp_transition = []
        tmp_SIR = [Event(np.random.uniform(0, l1), (N - k) * k / N), Event(l2, k)]
        for j in range(m):
            if i == j:
                tmp_transition.append(None)
            else:
                tmp_transition.append(Event(l3, N))
        transition_event.append(tmp_transition)
        SIR_event.append(tmp_SIR)

    return states, SIR_event, transition_event


def find_min(SIR_event, transition_event):
    min_time = float("inf")
    event = ""
    event_id = [0, 0]
    for i in range(len(SIR_event)):
        event_time = SIR_event[i][0].time()
        if min_time > event_time:
            min_time = event_time
            event = "SIR"
            event_id = [i, 0]

        event_time = SIR_event[i][1].time()
        if min_time > event_time:
            min_time = event_time
            event = "SIR"
            event_id = [i, 1]

    for i in range(len(transition_event)):
        if i == 0:
            event_time = transition_event[i][1].time()
        else:
            event_time = transition_event[i][0].time()
        if min_time > event_time:
            min_time = event_time
            event = "transition"
            k = np.random.randint(0, len(transition_event))
            while k == i:
                k = np.random.randint(0, len(transition_event))
            event_id = [i, k]
    return event, event_id


def update_state_event(states, SIR_event, transition_event, event, event_id):
    if event == "SIR":
        if event_id[1] == 0:
            states[event_id[0]].update("infected")
        else:
            states[event_id[0]].update("recover")
        SIR_event[event_id[0]][0].update(
            states[event_id[0]].s * states[event_id[0]].i / states[event_id[0]].N
        )
        SIR_event[event_id[0]][1].update(states[event_id[0]].i)
    else:
        person_out = states[event_id[0]].transit("out")
        states[event_id[1]].transit(person_out)
        for i in range(len(transition_event[0])):
            if transition_event[event_id[0]][i] is not None:
                transition_event[event_id[0]][i].update(states[event_id[0]].N)
            if transition_event[event_id[1]][i] is not None:
                transition_event[event_id[1]][i].update(states[event_id[1]].N)

    return states, SIR_event, transition_event


def format_result(states):
    for state in states:
        logger.info("s: %d, i: %d, r: %d, N: %d" % (state.s, state.i, state.r, state.N))
    logger.info("--------Terminated---------")


def print_info(event, event_id, states, status):
    if event == "SIR":
        logger.info(
            "%s %s: %d, %d, %d, %d"
            % (
                status,
                event,
                states[event_id[0]].s,
                states[event_id[0]].i,
                states[event_id[0]].r,
                states[event_id[0]].N,
            )
        )
    else:
        logger.info(
            "%s %s [MOVE OUT]: %d, %d, %d, %d [MOVE IN]: %d, %d, %d, %d"
            % (
                status,
                event,
                states[event_id[0]].s,
                states[event_id[0]].i,
                states[event_id[0]].r,
                states[event_id[0]].N,
                states[event_id[1]].s,
                states[event_id[1]].i,
                states[event_id[1]].r,
                states[event_id[1]].N,
            )
        )


def invalid(states):
    for state in states:
        if state.i > 2:
            return True 
    return False 

def read_args(infile):
    args = [None] * 6
    f = open(infile)
    inputs = f.readlines()

    for i in range(len(inputs)):
        args[i] = float(inputs[i])
    
    f.close()
    return int(args[0]), int(args[1]), int(args[2]), args[3], args[4], args[5]

def output_result(states, SIR_event, outfile):
    f = open(outfile+'.txt', 'w+') 

    lambda1 = []
    proportion = []
    for i in range(len(states)):
        f.write('%.4f\t%.4f\n' %(SIR_event[i][0].para, states[i].r / (states[i].r + states[i].s)))
        lambda1.append(SIR_event[i][0].para)
        proportion.append(states[i].r / (states[i].r + states[i].s))
    f.close()
    return lambda1, proportion

def plot_wrapper(x, y, m, N, k, l1, l2, l3, outfile):
    plt.figure(dpi = 300)
    plt.scatter(x, y, s=1)
    plt.xlabel('lambda 1')
    plt.ylabel('R/(R+S)')
    plt.title('m=%d,N=%d,k=10%d,l1=%d,l2=%d,l3=%.7f' %(m,N,k,l1,l2,l3 * m))
    plt.savefig(outfile)

def simulate(argv):
    infile = argv[1] # input file
    outfile = argv[2] # output file
    debug_info = False # debug logger mode
    if len(argv) > 3 and argv[3] == '-v':
        debug_info = True
    m, N, k, l1, l2, l3 = read_args(infile) # read args
    l3 = l3 / m
    # l3 = l3 / 100.0 # shrink value by 10
    states, SIR_event, transition_event = initialization(m, N, k, l1, l2, l3) # init
    while invalid(states): # simulate until no infections
        curr_event, curr_event_id = find_min(SIR_event, transition_event) # find min time event
        if debug_info:
            print_info(curr_event, curr_event_id, states, "before")
        states, SIR_event, transition_event = update_state_event(
            states, SIR_event, transition_event, curr_event, curr_event_id
        ) # update event
        if debug_info:
            print_info(curr_event, curr_event_id, states, "after")

    format_result(states) # format results for log
    lambda1, proportion = output_result(states, SIR_event, outfile) # output to output file
    plot_wrapper(lambda1, proportion, m, N, k, l1, l2, l3, outfile) # draw graphs

if __name__ == "__main__":
    simulate(sys.argv)

