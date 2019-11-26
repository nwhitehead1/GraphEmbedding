from pykg2vec.config.config import TransEConfig
from pykg2vec.config.config import TransEConfig, NTNConfig
from pykg2vec.core.TransE import TransE
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer
from pykg2vec.config.hyperparams import KGETuneArgParser


def transe_config(args, bayes_opt: bool = False):
    '''
    golden hyperparams
    self.learning_rate = 0.01
    self.L1_flag = True
    self.hidden_size = 50
    self.batch_size = 512
    self.epochs = 500
    self.margin = 1.0
    self.data = 'Freebase15k'
    self.optimizer = 'adam'
    self.sampling = "uniform"
    '''
    if bayes_opt:
        # Not sure what to do with this yet...
        golden = bayesian_optimization(args=args)

    return TransEConfig(args=args)


def ntn_config(args):
    '''
    Base configuration:
    self.learning_rate = 0.01
    self.L1_flag = True
    self.ent_hidden_size = 64
    self.rel_hidden_size = 32
    self.batch_size = 128
    self.epochs = 2
    self.margin = 1.0
    self.data = 'Freebase15k'
    self.optimizer = 'adam'
    self.sampling = "uniform"
    '''
    return NTNConfig(args=args)


def bayesian_optimization(args):
    bayes_opt = BaysOptimizer(args=args)
    bayes_opt.optimize()
    return bayes_opt.return_best()
