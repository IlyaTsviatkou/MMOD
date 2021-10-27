import numpy as np
import simpy

class Model:
    def __init__(self, _lambda, mu, v, m, n, env):

        self.__lambda = _lambda
        self.__mu = mu
        self.__v = v
        self.__m = m
        self.__n = n
        self.__env = env
        self.__request = simpy.Resource(env, n)
        self.__stat = []
        self.__queue_list = []
        self.__queue_time = []
        self.__total_requests = []
        self.__total_time = []
        self.__count = 0

        env.process(self.run())

    def add_request(self):
        self.__total_requests.append(len(self.__request.queue) + self.__request.count)
        self.__queue_list.append(len(self.__request.queue))
        with self.__request.request() as Request:
            self.__count += 1
            queque = len(self.__request.queue)
            active_channel = self.__request.count
            self.__stat.append(queque + active_channel - 1)
            if queque <= self.__m:
                print("Request {0} sent for processing : {1}".format(self.__count, self.__env.now))
                t1 = self.__env.timeout(np.random.exponential(1 / self.__v), value = 'reject')
                time_in = self.__env.now
                res = yield Request | t1
                self.__queue_time.append(self.__env.now - time_in)
                if res == {t1: 'reject'}:
                    print("Request {0} rejected (Waiting time in queue exceeded) at {1}".format(self.__count, self.__env.now))
                else:
                    yield self.__env.process(self.service())
                    print("Request {0} done at {1}".format(self.__count, self.__env.now))
                self.__total_time.append(self.__env.now - time_in)
            else:
                self.__queue_time.append(0)
                self.__total_time.append(0)
                print("Request {0} rejected(Queue size exceeded) at {1}".format(self.__count, self.__env.now))

    def run(self):
        while True:
            yield self.__env.timeout(np.random.exponential(1 / self.__lambda))
            self.__env.process(self.add_request())

    def service(self):
        yield self.__env.timeout(np.random.exponential(1 / self.__mu))
            
    def get_data(self):
        return self.__stat, self.__queue_list, self.__total_requests, self.__queue_time, self.__total_time