import simpy
import Model
import Statistics

if __name__ == '__main__':
    time = 100
    ######## values
    _lambda = 10
    mu = 3
    v = 6
    m = 4   
    n = 3
    ########

    env = simpy.Environment()
    model = Model.Model(_lambda, mu, v, m, n, env)
    env.run(time)

    Statistic = Statistics.Statistics(_lambda, mu, v, m, n, model.get_data())
    Statistic.sustainability_test(2,3,10,3,6,100)
    Statistic.sustainability_test(3,2,8,4,5,200)
    Statistic.sustainability_test(4,1,6,2,4,300)
    #Statistic.generate()
    Statistic.steady_test()
    
                    