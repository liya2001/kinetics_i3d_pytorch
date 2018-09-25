import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str, help='data-set')

parser.add_argument('--num_classes', type=int, default=13, help='num classes')
