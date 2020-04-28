import argparse
import os
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def path(p):
    return os.path.expanduser(p)

PATH = 'CHANGE ME PLEASE'

parser = argparse.ArgumentParser(description='Arguments of DeepDIG project')
parser.add_argument("--project-dir",default=PATH)
parser.add_argument("--dataset",default='MNIST')
parser.add_argument("--pre-trained-model",default='CNN')
parser.add_argument("--dropout", type=float, required=False, default=0.0, help="Ratio of dropout")
parser.add_argument("--lr", type=float, required=False, default=0.01, help="Learning rate")
parser.add_argument("--step-size-scheduler", type=int, required=False, default=1000, help="The learning rate step size scheduler")
parser.add_argument("--gamma-scheduler", type=float, required=False, default=0.95, help="Gamma of scheduler")
parser.add_argument('--cuda', type=str2bool, default=True, help='enables CUDA training')
parser.add_argument('--steps', type=int, default=5000,
                    help='number of steps to train (default: 2000)')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')

parser.add_argument("--middle-point-threshold", type=float, required=False, default=0.0001,
                    help="Coefficient of reconstruction loss")
parser.add_argument("--alpha", type=float, required=False, default=0.8,
                    help="Coefficient of target loss")
parser.add_argument("--classes",type=str,default="1;2",help="The investigated classes")
parser.add_argument('--save_samples', type=str2bool, default=True,
                    help='enables saving generated samples')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--pre-trained-model-input-shape', type=str, default="1;28;28",
                    help='shape of the input data to pre trained model')
parser.add_argument("--num-samples-trajectory", type=int, required=False, default=50,
                    help="Number of samples generated in the trajectory line between x(t)=t*x0+(1-t)*x1")

args = parser.parse_args()
