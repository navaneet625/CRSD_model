# simple runner
import argparse
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='experiments/exp_crsd_tiny.yaml')
    args = parser.parse_args()
    train(args.config)