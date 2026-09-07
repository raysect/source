# Copyright (c) 2014-2026, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from multiprocessing import get_context, cpu_count
import random
import time
from .base import RenderEngine


class MulticoreEngine(RenderEngine):
    """
    A render engine for distributing work across multiple CPU cores.

    The number of processes spawned by this render engine is controlled via
    the processes attribute. This can also be set at object initialisation.

    If the processes attribute is set to None (the default), the render engine
    will automatically set the number of processes to be equal to the number
    of CPU cores detected on the machine.

    If a render is being performed where the time to compute an individual task
    is comparable to the latency of the inter process communication (IPC), the
    render may run significantly slower than expected due to waiting for the
    IPC to complete. To reduce the impact of the IPC overhead, multiple tasks
    are grouped together into jobs, requiring only one IPC wait for multiple
    tasks.

    By default the number of tasks per job is adjusted automatically. The
    tasks_per_job attribute can be used to override this automatic adjustment.
    To reenable the automated adjustment, set the tasks_per_job attribute to
    None.

    :param processes: The number of worker processes, or None to use all available cores (default).
    :param tasks_per_job: The number of tasks to group into a single job, or None if this should be determined automatically (default).
    :param start_method: The method used to start child processes: 'fork' (default), 'spawn' or 'forkserver'.

    .. code-block:: pycon

        >>> from raysect.core import MulticoreEngine
        >>> from raysect.optical.observer import PinholeCamera
        >>>
        >>> camera = PinholeCamera((512, 512))
        >>>
        >>> # allowing the camera to use all available CPU cores.
        >>> camera.render_engine = MulticoreEngine()
        >>>
        >>> # or forcing the render engine to use a specific number of CPU processes
        >>> camera.render_engine = MulticoreEngine(processes=8)
    """

    def __init__(self, processes=None, tasks_per_job=None, start_method='fork'):
        super().__init__()
        self.processes = processes
        self.tasks_per_job = tasks_per_job
        self._context = get_context(start_method)

    @property
    def processes(self):
        return self._processes

    @processes.setter
    def processes(self, value):
        if value is None:
            self._processes = cpu_count()
        else:
            value = int(value)
            if value <= 0:
                raise ValueError('Number of concurrent worker processes must be greater than zero.')
            self._processes = value

    @property
    def tasks_per_job(self):
        return self._tasks_per_job

    @tasks_per_job.setter
    def tasks_per_job(self, value):
        if value is None:
            self._tasks_per_job = 1
            self._auto_tasks_per_job = True
        else:
            if value < 1:
                raise ValueError("The number of tasks per job must be greater than zero or None.")
            self._tasks_per_job = value
            self._auto_tasks_per_job = False

    def run(self, tasks, render, update, render_args=(), render_kwargs={}, update_args=(), update_kwargs={}):

        # establish ipc queues
        job_queue = self._context.SimpleQueue()
        result_queue = self._context.SimpleQueue()
        tasks_per_job = self._context.Value('i')

        # start process to generate jobs
        tasks_per_job.value = self._tasks_per_job
        producer = self._context.Process(target=self._producer, args=(tasks, job_queue, tasks_per_job))
        producer.start()

        # start worker processes
        workers = []
        for pid in range(self._processes):
            p = self._context.Process(target=self._worker, args=(render, render_args, render_kwargs, job_queue, result_queue))
            p.start()
            workers.append(p)

        # consume results
        remaining = len(tasks)
        while remaining:

            results = result_queue.get()

            # has a worker failed?
            if isinstance(results, Exception):

                # clean up
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                producer.terminate()

                # wait for processes to terminate
                for worker in workers:
                    worker.join()
                producer.join()

                # raise the exception to inform the user
                raise results

            # update state with new results
            for result in results:
                update(result, *update_args, **update_kwargs)
                remaining -= 1

        # shutdown workers
        for _ in workers:
            job_queue.put(None)

        # store tasks per job value for next run
        self._tasks_per_job = tasks_per_job.value

    def worker_count(self):
        return self._processes

    def _producer(self, tasks, job_queue, stored_tasks_per_job):

        # initialise request rate controller constants
        target_rate = 50  # requests per second
        min_time = 1      # seconds
        min_requests = min(2 * target_rate, 5 * self._processes)
        tasks_per_job = stored_tasks_per_job.value

        # split tasks into jobs and dispatch to workers
        requests = -self.processes  # ignore the initial jobs, the requests are instantaneous
        start_time = time.time()
        while tasks:

            # assemble job
            job = []
            for _ in range(tasks_per_job):
                if tasks:
                    job.append(tasks.pop())
                    continue
                break

            # add job to queue
            job_queue.put(job)
            requests += 1

            # if enabled, auto adjust tasks per job to keep target requests per second
            if self._auto_tasks_per_job:

                elapsed_time = (time.time() - start_time)
                if elapsed_time > min_time and requests > min_requests:

                    # re-normalise the tasks per job based on previous work to propose a new value
                    requests_rate = requests / elapsed_time
                    proposed = tasks_per_job * requests_rate / target_rate

                    # gradually adjust tasks per job to reduce risk of oscillation
                    tasks_per_job = 0.1 * proposed + 0.9 * tasks_per_job
                    tasks_per_job = max(1, round(tasks_per_job))

                    # reset counters
                    requests = 0
                    start_time = time.time()

        # pass back new value
        stored_tasks_per_job.value = tasks_per_job

    def _worker(self, render, args, kwargs, job_queue, result_queue):

        # re-seed the random number generator to prevent all workers inheriting the same sequence
        random.seed()

        # process jobs
        while True:

            job = job_queue.get()

            # have we been commanded to shutdown?
            if job is None:
                break

            results = []
            for task in job:
                try:
                    results.append(render(task, *args, **kwargs))
                except Exception as e:
                    # pass the exception back to the main process and quit
                    result_queue.put(e)
                    break

            # hand back results
            result_queue.put(results)
