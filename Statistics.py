import math
import numpy as np
from scipy.stats.stats import mode
import simpy
import Model
import Statistics


import matplotlib.pyplot as plt
from scipy.stats import chisquare


class Statistics:

    def __init__(self, _lambda, mu, v, m, n, data):
        self.__lambda = _lambda
        self.__mu = mu
        self.__v = v
        self.__m = m
        self.__n = n
        self.__stat = np.array(data[0])
        self.__queue_list = np.array(data[1])
        self.__total_requests = np.array(data[2])
        self.__queue_time = np.array(data[3])
        self.__total_time = np.array(data[4])

    def show_chart(self, values , title):
        X = range(self.__m + self.__n + 1)
        fig, ax = plt.subplots(1, 1)
        ax.bar(X, values, width=0.1)
        ax.set_title(title+'_Histogram')
        # plt.figure()

    ##### E
    def get_e_prob(self):
        P = [len(self.__stat[self.__stat == index]) / len(self.__stat) for index in range(self.__n + self.__m + 1)]
        for index, p in enumerate(P):
            print('p{0}: {1}'.format(index, p))
        # self.show_chart(P)
        return len(self.__stat[self.__stat == 0]) / len(self.__stat), len(
            self.__stat[self.__stat == self.__m + self.__n]) / len(self.__stat)

    def get_e(self):
        return [len(self.__stat[self.__stat == index]) / len(self.__stat) for index in range(self.__n + self.__m + 1)]

    def avg_gueqe_e(self):
        return self.__queue_list.mean()

    def avg_total_e(self):
        return self.__total_requests.mean()

    def avg_gueqe_time_e(self):
        return self.__queue_time.mean()

    def avg_total_time_e(self):
        return self.__total_time.mean()
    #####

    ##### T    
    def get_t_prob(self):
        teta = self.__lambda / self.__mu
        beta = self.__v / self.__mu
        p0 = 1 / (sum([(teta ** index) / math.factorial(index) for index in range(self.__n + 1)]) + (
                teta ** self.__n) / math.factorial(self.__n) * sum(
            [(teta ** index) / np.prod(np.array([self.__n + l * beta for l in range(1, index + 1)])) for index in
             range(1, self.__m + 1)]))
        print('\np0: {0}'.format(p0))
        for index in range(1, self.__n + 1):
            print('p{0}: {1}'.format(index, (p0 * teta ** index) / math.factorial(index)))      
        pn = (p0 * teta ** self.__n) / math.factorial(self.__n)
        for index in range(1, self.__m + 1):
            print('p{0}: {1}'.format(self.__m + index - 1, pn * (teta ** index) / np.prod(
                np.array([self.__n + l * beta for l in range(1, index + 1)]))))
        pot = pn * (teta ** self.__m) / np.prod(np.array([self.__n + l * beta for l in range(1, self.__m + 1)]))
        return round(p0, 15), round(pot, 15), self.avg_gueqe_t(pn), self.avg_total_t(p0, pn)

    def get_t(self):
        teta = self.__lambda / self.__mu
        beta = self.__v / self.__mu
        p = []
        p.append(1 / (sum([(teta ** index) / math.factorial(index) for index in range(self.__n + 1)]) + (
                teta ** self.__n) / math.factorial(self.__n) * sum(
            [(teta ** index) / np.prod(np.array([self.__n + l * beta for l in range(1, index + 1)])) for index in
             range(1, self.__m + 1)])))
        for index in range(1, self.__n + 1):
            p.append((p[0] * teta ** index) / math.factorial(index))
        pn = p[-1]
        for index in range(1, self.__m + 1):
            p.append(pn * (teta ** index) / np.prod(
                np.array([self.__n + l * beta for l in range(1, index + 1)])))
        return p

    def avg_gueqe_t(self, pn):
        teta = self.__lambda / self.__mu
        beta = self.__v / self.__mu
        return sum(
            [index * pn * (teta ** index) / np.prod(np.array([self.__n + l * beta for l in range(1, index + 1)])) for
             index in range(1, self.__m + 1)])

    def avg_total_t(self, p0, pn):
        teta = self.__lambda / self.__mu
        beta = self.__v / self.__mu

        return sum([index * p0 * (teta ** index) / math.factorial(index) for index in range(1, self.__n + 1)]) + sum(
            [(self.__n + index) * pn * teta ** index / np.prod(
                np.array([self.__n + l * beta for l in range(1, index + 1)])) for
             index in range(1, self.__m + 1)])
    #####

    def test_queuing_system(n, m, _lambda, mu, v, time):
        env = simpy.Environment()
        model = Model.Model(_lambda, mu, v, m, n, env)
        env.run(time)
        return model.get_data()

    def steady_test(self):
        n, m, lambd, mu, v = 2, 1, 1 , 1, 1
        p = []
        times = [i*30 for i in range(1,50)]
        p0, p1, p2, p3 = [],[],[],[]
        teta = lambd / mu
        beta = v / mu
        p = []
        p.append(1 / (sum([(teta ** index) / math.factorial(index) for index in range(n + 1)]) + (
                teta ** n) / math.factorial(n) * sum(
            [(teta ** index) / np.prod(np.array([n + l * beta for l in range(1, index + 1)])) for index in
             range(1, m + 1)])))
        for index in range(1, n + 1):
            p.append((p[0] * teta ** index) / math.factorial(index))
        pn = p[-1]
        for index in range(1, m + 1):
            p.append(pn * (teta ** index) / np.prod(
                np.array([n + l * beta for l in range(1, index + 1)])))
        
        for time in times:
            env = simpy.Environment()
            model = Model.Model(lambd, mu, v, m, n, env)
            env.run(time)
            test_results = model.get_data()
            __stat = np.array(test_results[0])
            __queue_list = np.array(test_results[1])
            __total_requests = np.array(test_results[2])
            __queue_time = np.array(test_results[3])
            __total_time = np.array(test_results[4])
            empirical_characteristic = [len(__stat[__stat == index]) / len(__stat) for index in range(n + m + 1)]
            p0.append(empirical_characteristic[0])
            p1.append(empirical_characteristic[1])
            p2.append(empirical_characteristic[2])
            p3.append(empirical_characteristic[3])
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=4, figsize=(16, 28), 
                                         gridspec_kw={'height_ratios': [1, 1, 1, 1], 'width_ratios': [1]}
                                        )
        x = np.linspace(0, 10, 10000)

        ax0.set_title('p0')
        ax0.plot(times, [p[0]]*len(times), '-k')
        ax0.fill_between(times, y1=p0, color='b', step='post', alpha=0.5)

        ax1.set_title('p1')
        ax1.plot(times, [p[1]]*len(times), '-k')
        ax1.fill_between(times, y1=p1, color='r', step='post', alpha=0.5)


        ax2.set_title('p2')
        ax2.plot(times, [p[2]]*len(times), '-k')
        ax2.fill_between(times, y1=p2, color='g', step='post', alpha=0.5)


        ax3.set_title('p3')
        ax3.plot(times, [p[3]]*len(times), '-k')
        ax3.fill_between(times, y1=p3, color='y', step='post', alpha=0.5)

        # fig.tight_layout()
        plt.show()

    def sustainability_test(self,n, m, lambd, mu, v,time):
        #n, m, lambd, mu, v = 2, 1, 1 , 1, 1
        p = []
        teta = lambd / mu
        beta = v / mu
        p = []
        p.append(1 / (sum([(teta ** index) / math.factorial(index) for index in range(n + 1)]) + (
                teta ** n) / math.factorial(n) * sum(
            [(teta ** index) / np.prod(np.array([n + l * beta for l in range(1, index + 1)])) for index in
             range(1, m + 1)])))
        for index in range(1, n + 1):
            p.append((p[0] * teta ** index) / math.factorial(index))
        pn = p[-1]
        for index in range(1, m + 1):
            p.append(pn * (teta ** index) / np.prod(
                np.array([n + l * beta for l in range(1, index + 1)])))

        env = simpy.Environment()
        model = Model.Model(lambd, mu, v, m, n, env)
        env.run(time)
        test_results = model.get_data()
        __stat = np.array(test_results[0])
        __queue_list = np.array(test_results[1])
        __total_requests = np.array(test_results[2])
        __queue_time = np.array(test_results[3])
        __total_time = np.array(test_results[4])
        empirical_characteristic = [len(__stat[__stat == index]) / len(__stat) for index in range(n + m + 1)]
        X = range(m + n + 1)
        fig,ax2 = plt.subplots(1, 1)
        fig,ax1 = plt.subplots(1, 1)
        ax1.bar(X, p, width=0.1)
        ax1.set_title('Theoretical_Histogram') 
        ax2.bar(X, empirical_characteristic, width=0.1)
        ax2.set_title('Empirical_Histogram')  
        print(chisquare(p, empirical_characteristic))
        plt.show()


    # STATS
    def generate(self):
        print('\n')
        print('The intensity of request\'s flow :{}'.format(self.__lambda))
        print('The intensity rate flow : {}'.format(self.__mu))
        print('Time of request\'s life in the queue : {}'.format(self.__v))
        print('Queue size : {}'.format(self.__m))
        print('Quantity of channels :{}\n'.format(self.__n))

        ##### Empirical
        e_prob = self.get_e_prob()
        print('\n')
        print('Empirical probability of failure : {0}'.format(e_prob[1]))
        print("Empirical p0: {}".format(e_prob[0]))
        Q_e = 1 - e_prob[1]
        print("Empirical relative bandwidth : {}".format(Q_e))
        A_e = Q_e * self.__lambda
        print("Empirical absolute bandwidth : {}".format(A_e))
        print("Empirical average number of queued requests : {}".format(self.avg_gueqe_e()))
        print("Empirical average number of requests served in the QS : {}".format(self.avg_total_e()))
        avg_off_e = Q_e * self.__lambda / self.__mu
        print("Empirical average occupied channels : ", avg_off_e)
        print("Empirical average time of a request being in the queue: ", self.avg_gueqe_time_e())
        print("Empirical average time of a request being in the queue: ", self.avg_total_time_e())
        #####

        ##### Theoretical
        t_prob = self.get_t_prob()
        print('\n')
        print('Theoretical probability of failure :{0}'.format(t_prob[1]))
        print("Theoretical p0 : {}".format(t_prob[0]))
        Q_t = 1 - t_prob[1]
        print("Theoretical relative bandwidth : {}".format(Q_t))
        A_t = Q_t * self.__lambda
        print("Theoretical absolute bandwidth : {}".format(A_t))
        print("Theoretical average number of queued requests : {}".format(t_prob[2]))
        print("Theoretical average number of applications served in the QS : {}".format(t_prob[3]))
        avg_off_t = Q_t * self.__lambda / self.__mu
        print("Theoretical average occupied channels : ", avg_off_t)
        avg_queque_t = t_prob[2] / self.__lambda
        print("Theoretical average time an request beeing is in the queue : ", avg_queque_t)
        avg_SMO_t = t_prob[3] / self.__lambda
        print("Theoretical average time an request beeing is in the queue : ", avg_SMO_t)
        #####

        E_ver = self.get_e()
        P_ver = self.get_t()
        self.show_chart(self.get_t(),"Theoretical")
        self.show_chart(self.get_e(),"Empirical")
        # plt.show()
        print(chisquare(E_ver, P_ver))
        plt.show()
        #self.steady_test()

    
    