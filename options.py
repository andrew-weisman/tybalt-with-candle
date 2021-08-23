import os, sys
import argparse

def parse_training_arguments(model_name):
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    parser.add_argument('-d', '--depth', dest="depth", type=str, 
        help='Number of layers between input and latent layer; default=1',
            metavar="depth",  default="1")
    parser.add_argument('--hidden_dim', dest="hidden_dim",
        help='Dimensionality of the first hidden layer',
            metavar="hidden_dim", type=str, default='300')
    parser.add_argument('-k', '--kappa', type=str, dest="kappa", 
        help='How fast to linearly ramp up KL loss', 
            metavar="kappa", default='1.')
    parser.add_argument("-m", "--model_name", dest="model_name", 
        help="model name: adage | vae", metavar="model_name", default=model_name)
    parser.add_argument('-N', '--noise', type=float, dest="noise",
        help='How much Gaussian noise to add during training',
            metavar="noise", default=0.05)
    parser.add_argument('-s', '--sparsity', dest="sparsity", type=float,
        help='How much L1 regularization penalty to apply',
            metavar="sparsity", default=0.001)
    parser.add_argument("-v", "--verbose", action="store_true",dest="verbose",
        help="increase the verbosity level of output",default=False)
    parser.add_argument('-z', '--latent_dim', dest="latent_dim", default=100, 
        help='Dimensionality of the latent space', metavar="latent_dim")
    parser.add_argument("-b", "--bs", dest="batch_size", type=str, \
        help="batch size; default=50", metavar="batch_size", default='50')
    parser.add_argument("-E", "--executions_per_trial", dest="ex_per_trial",
        help="# of executions per HPO trial; default=3", type=int,
            metavar="executions_per_trial", default=3)
    parser.add_argument('-f', '--output_filename', dest="output_filename",
        help='The name of the file to store results',
            metavar="output_filename")
    parser.add_argument("--hpo", dest="hpo", type=str, default=None,
        help="hpo method: ktuner-random | ktuner-bayesian | ktuner-hyperband " \
                       + "| candle-grid | candle-bayesian, " \
                       + " default=None", metavar="hpo_method")
    parser.add_argument("-l", "--lr", dest="lr", type=str,
        help="learning rate; default=0.0005", metavar="learning_rate",
            default="1.e-5")
    parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=str,
        help="num # of epochs; default=100",
            metavar="num_epochs", default="100")
    parser.add_argument("--max_trials", dest="max_trials", type=int,
        help="max # of trials in hp optimization; default=50",
            metavar="num_trials", default=50)
    parser.add_argument("-o", "--optimizer", dest="optimizer",
        help="optimizer: adam | rmsprop ", metavar="optimizer", default="adam")
    parser.add_argument("--val_split",dest="val_split", type=float,
        help="increase the verbosity level of output",
            metavar="val_split", default=0.2)
    parser.add_argument('-W', '--wall_time', dest="wall_time", type=str,
        help='Wall time for HPO with CANDLE; default=\"24:00:00\"',
            metavar="wall_time",default="24:00:00")
    args = parser.parse_args()
    return args

def parse_prediction_arguments():
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group('required arguments')
    parser.add_argument('-d', '--depth', dest="depth", type=str,
        help='Number of layers between input and latent layer; default=1',
            metavar="depth",  default="1")
    parser.add_argument('--hidden_dim', dest="hidden_dim",
        help='Dimensionality of the first hidden layer',
            metavar="hidden_dim", type=str, default='300')
    parser.add_argument('-k', '--kappa', type=str, dest="kappa",
        help='How fast to linearly ramp up KL loss',
            metavar="kappa", default='1.')
    required_args.add_argument("-m", "--model_name", dest="model_name",
        help="model name: adage | vae", metavar="model_name")
    parser.add_argument('-N', '--noise', type=float, dest="noise",
        help='How much Gaussian noise to add during training',
            metavar="noise", default=0.05)
    parser.add_argument("-o", "--output", type=str, dest="test_results",
        help="output file with test results; default='test_results.h5'",
            metavar="test_results", default="test_results.h5")
    parser.add_argument('-s', '--sparsity', dest="sparsity", type=float,
        help='How much L1 regularization penalty to apply',
            metavar="sparsity", default=0.001)
    parser.add_argument("-v", "--verbose", action="store_true",dest="verbose",
        help="increase the verbosity level of output",default=False)
    parser.add_argument('-z', '--latent_dim', dest="latent_dim", default=100,
        help='Dimensionality of the latent space', metavar="latent_dim")
    args = parser.parse_args()
    return args

def parse_command_line_arguments(task, model_name):
    if task == "predict":
        opt = parse_prediction_arguments()
        checkpoint_combined = 'checkpoints/' + ".".join(["tybalt",   
                              'combined', opt.model_name, 'h5'])
        return (opt, checkpoint_combined)

    elif task == "train":
        # task == "train"
        opt = parse_training_arguments(model_name)
        if not os.path.isdir(os.path.join(os.getcwd(), 'checkpoints')):
           os.mkdir(os.path.join(os.getcwd(), 'checkpoints'))
        if not opt.model_name:
            sys.exit("Please specify a model name")

        if not opt.hpo in ["random", "bayesian", "hyperband", None]:
            sys.exit("\nUnsupported HPO method: " + opt.hpo)

        checkpoint_combined = 'checkpoints/' + ".".join(["tybalt",  
                              'combined', opt.model_name, 'h5'])
        checkpoint_encoder  = 'checkpoints/' + ".".join(["tybalt",   \
                              'encoder', opt.model_name, 'h5'])
        checkpoint_decoder  = 'checkpoints/' + ".".join(["tybalt",   \
                          'decoder', opt.model_name, 'h5'])
        return (opt, checkpoint_combined, checkpoint_encoder, checkpoint_decoder)
    else:
        sys.exit("Unsupported task: " + task)
