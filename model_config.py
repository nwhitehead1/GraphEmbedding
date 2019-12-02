from pykg2vec.config.config import TransEConfig
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer


def transe_config(args, bayes_opt: bool = False):
    args.learning_rate = 0.01
    args.epochs = 200
    args.margin = 2.0
    args.batch_training = 256

    if bayes_opt:
        # Not sure what to do with this yet...
        # Currently using -ghp arg - only implemented for Freebase15k!
        golden = bayesian_optimization(args=args)

    return TransEConfig(args=args)

def bayesian_optimization(args):
    bayes_opt = BaysOptimizer(args=args)
    bayes_opt.optimize()
    return bayes_opt.return_best()
