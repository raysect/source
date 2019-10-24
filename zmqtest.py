import zmq
import time
import multiprocessing


def worker():

    ctx = zmq.Context()

    work = ctx.socket(zmq.PULL)
    # work.connect('ipc://workq.ipc')
    work.connect('tcp://127.0.0.1:3500')

    while True:
        v = work.recv_pyobj()
        if v == 'SHUTDOWN':
            break
        # print(v)


ctx = zmq.Context()
wrk = ctx.socket(zmq.PUSH)
# wrk.bind('ipc://workq.ipc')
wrk.bind('tcp://127.0.0.1:3500')

p = multiprocessing.Process(target=worker)
p.start()

for i in range(1000000):
    if not i % 100000: print('sending {}...'.format(i))
    wrk.send_pyobj('{}'.format(i))
wrk.send_pyobj('SHUTDOWN')



