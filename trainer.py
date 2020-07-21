from pytorch_lightning import Trainer
from argparse import ArgumentParser
from omegaconf import OmegaConf
from pathlib import Path

from meshgen_utils import utils
from models import GCN

project_root = Path(__file__).absolute().parent

def main(args):
    hp = OmegaConf.load(args.config)
    print(args)
    print(hp)
    print(project_root)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')

    args = parser.parse_args()
    main(args)