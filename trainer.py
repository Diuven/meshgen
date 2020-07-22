from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from omegaconf import OmegaConf
from pathlib import Path

from meshgen_utils import utils
from models import GCN

project_root = Path(__file__).absolute().parent

def main(args):
    hp = OmegaConf.load(args.config)

    train_name = "GCN_%s" % (Path(hp.data.file).stem)
    logger = TensorBoardLogger('logs/', name=train_name)
    logger.log_hyperparams(OmegaConf.to_container(hp))

    net = GCN(hp)

    trainer = Trainer(logger=logger, max_epochs=args.max_epochs, gpus=-1)
    trainer.fit(net)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--max_epochs', type=int, default=10000)
    # checkpoint

    args = parser.parse_args()
    main(args)