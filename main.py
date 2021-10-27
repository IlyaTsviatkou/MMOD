import simpy
import Model
import Statistics

if __name__ == '__main__':
    time = 1000
    ######## values
    _lambda = 8
    mu = 3
    v = 4
    m = 2
    n = 3
    ########

    env = simpy.Environment()
    model = Model.Model(_lambda, mu, v, m, n, env)
    env.run(time)

    Statistic = Statistics.Statistics(_lambda, mu, v, m, n, model.get_data())
    Statistic.generate()