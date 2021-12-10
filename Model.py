import numpy as np
import simpy

class Model:
    def __init__(self, _lambda, mu, v, n, env):

        self.__lambda = _lambda
        self.__mu = mu
        self.__v = v
        self.__n = n
        self.__env = env
        self.__request = simpy.Resource(env, n)
        self.__stat = []
        self.__total_requests = []
        self.__total_time = []
        self.__count = 0

        env.process(self.run())

    def add_request(self):
        self.__total_requests.append(self.__request.count)
        with self.__request.request() as Request:
            self.__count += 1
            active_channel = self.__request.count

            self.__stat.append(active_channel - 1)
            print("Request {0} sent for processing : {1}".format(self.__count, self.__env.now))
            t1 = self.__env.timeout(0, value = 'reject')
            time_in = self.__env.now
            res = yield Request | t1
            if res == {t1: 'reject'}:
                print("Request {0} rejected at {1}".format(self.__count, self.__env.now))
                self.__total_time.append(0)
            else:
                yield self.__env.process(self.service())
                print("Request {0} done at {1}".format(self.__count, self.__env.now))
                self.__total_time.append(self.__env.now - time_in)
        

    def run(self):
        while True:
            yield self.__env.timeout(np.random.exponential(1 / self.__lambda))
            self.__env.process(self.add_request())

    def service(self):
        yield self.__env.timeout(np.random.exponential(1 / self.__mu) + np.random.exponential(1/self.__v))
            
    def get_data(self):
        return self.__stat, self.__total_requests, self.__total_time