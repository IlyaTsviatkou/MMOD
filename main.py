import simpy
import Model
import Statistics

if __name__ == '__main__':
    time = 100
    ######## values
    _lambda = 2
    mu = 3
    v = 6   
    n = 3
    ########

    env = simpy.Environment()
    model = Model.Model(_lambda, mu, v, n, env)
    env.run(time)

    Statistic = Statistics.Statistics(_lambda, mu, v, n, model.get_data())
    Statistic.generate()
   
    
                    