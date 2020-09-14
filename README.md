# LMPC Examples

This repo collects a few LMPC examples. In particular we implemented:
- The linear LMPC to solve the constrainted linear quadratic regulator (folder LinearMPC)
- The nonlinear LMPC to solve nonlinear constrainted minimum time problems and simple racing problems (folder Nonlinear LMPC). 
- The decentralized nonlinaer LMPC for cooperative control of dinamically decoupled systems coupled by feasibility constraints (folder CoopLMPC)

For more details please check the README file in each folder

## Native/VirtualEnv Installation

The code is written in Python 2.7 and the following packages are required

```
apt install coinor-libipopt-dev
pip install cvxpy
apt install -y libblas-dev
pip install ipopt
pip intall casadi
```

## Docker Installation

We also provide Dockerfiles and scripts to build a Docker image and instantiate containers to run the code in this repository. To do so, first we need to [install Docker CE](https://docs.docker.com/install/). Instructions for installing via the `apt` repository for [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/) are included below. Instructions for MacOS can be found [here](https://docs.docker.com/docker-for-mac/install/).

___Ubuntu___

```
# Update apt package index
sudo apt update

# Install packages to allow apt to use a repository over HTTPS
sudo apt install -y apt-transport-https \
  ca-certificates \
  curl \
  software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add Docker repository
sudo add-apt-repository \
  "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) \
  stable"

# Update apt package index
sudo apt update

# Install Docker CE
sudo apt install docker-ce

# Add user to docker group
export user=$(whoami)
sudo usermod -aG docker $user
```

Docker related files are included in the `docker` directory. The bash script `start_LMPC_docker.sh` builds the image, which extends the `continuumio/anaconda` image by adding the cvxpy and ipopt Python libraries.

### Running with Docker

In order to (re)build the image and start the container, run the following command from the base directory of the repository:

```
bash docker/start_LMPC_docker.sh -r
```

The `-r` flag triggers a rebuild of the image. There is no harm in always including this flag to ensure that changes to the Dockerfile are implemented. If the `-r` flag is not specified, a container will be instantiated using the most recent build of the image.

Once the container is running, open up a new terminal and use the following command to start a bash session in the container (you can get the name of the container using `docker container ls`):

```
docker exec -it ${container_name} bash
```

We can then run our code within this environment.

#### Important Notes

- Upon container instantiation the base directory of the repository is mounted to `/home/LMPC` within the container
- `python-tk` is installed and `TkAgg` needs to be selected as the `matplotlib` backend as opposed to the default `Qt5Agg` (which doesn't seem to work). This can be done by including the following code once __before__ the first time `matplotlib.pyplot` is imported:
```
import matplotlib
matplotlib.use('TkAgg')
```
- A Jupyter notebook server can be started within the container using the command:
```
/opt/conda/bin/jupyter notebook --notebook-dir=/home/LMPC --ip=0.0.0.0 --port=8888 --allow-root
```
Once it is up and running, enter the following URL into a web browser:
```
localhost:8888/?token=[server_token]
```
where `server_token` can be found in the terminal print-out after starting the notebook server
