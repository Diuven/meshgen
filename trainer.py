from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
from omegaconf import OmegaConf
from pathlib import Path
import os

from meshgen_utils import utils
from models import MeshRefineGCN, MeshRefinePN, CondenseMesh

project_root = Path(__file__).absolute().parent

def main(args):
    hp = OmegaConf.load(args.config)

    train_name = "%s_%s" % (hp.model.name.upper(), Path(hp.data.file).stem)
    logger = TensorBoardLogger('logs/', name=train_name)
    logger.log_hyperparams(OmegaConf.to_container(hp))

    if hp.model.name == 'gcn':
        net = MeshRefineGCN(hp)
    elif hp.model.name == 'pn':
        net = MeshRefinePN(hp)
    elif hp.model.name == 'zero':
        net = CondenseMesh(hp)
    else:
        raise ValueError("Invalid model name: %s" % hp.model.name)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        gpus=-1,
        default_root_dir='logs')
    trainer.fit(net)
    
    mesh, pcd = net.current_mesh, net.source_pcd
    if hp.model.name != 'zero':
        net.get_loss.show(mesh)
    utils.show_overlay(mesh, pcd)
    utils.save_result(os.path.join(logger.log_dir, 'objects'), -1, mesh, pcd)
    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/test.yaml')
    parser.add_argument('--max_epochs', type=int, default=10000)
    # checkpoint

    args = parser.parse_args()
    main(args)