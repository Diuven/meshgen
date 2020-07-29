# Point Cloud to Mesh

## How to install

clone the repo.
edit config at `config/default.yaml`

### Docker

run `docker build . -t diuven/meshgen:latest`

mount current directory, and run `python3 trainer.py` in docker container

### venv

run `pip install -f https://pytorch-geometric.com/whl/torch-1.4.0.html -r requirements.txt`.

run

```sh
wget https://anaconda.org/pytorch3d/pytorch3d/0.2.0/download/linux-64/pytorch3d-0.2.0-py37_cu101_pyt14.tar.bz2 \
&& conda install ./pytorch3d-0.2.0-py37_cu101_pyt14.tar.bz2 \
&& rm ./pytorch3d-0.2.0-py37_cu101_pyt14.tar.bz2
```

activate venv, and run `python3 trainer.py`
