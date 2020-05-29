# Start from python image
FROM python:2.7.16

# Install python dev and latex libraries
RUN apt update
RUN apt install -y texlive-fonts-recommended texlive-fonts-extra dvipng
RUN apt install -y python-dev python-tk
RUN apt install -y libmpfr-dev libmpc-dev libppl-dev coinor-libipopt-dev

# Set display environment variable
ENV QT_X11_NO_MITSHM=1

# Set up home directory in image (this is where the repo directory will be mounted)
RUN mkdir /home/LMPC
WORKDIR /home/LMPC

RUN pip install --upgrade pip
RUN pip install cython cysignals gmpy2==2.1.0b3
RUN pip install numpy scipy matplotlib jupyterlab
RUN pip install cvxpy pplpy
RUN apt install -y libblas-dev
RUN pip install ipopt
RUN pip install casadi
RUN pip install scikit-learn

RUN apt install -y ffmpeg

RUN pip install pypoman

# Two options for command to run

# Start jupyter notebook server
CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root

# Keep container open after creation
# CMD tail -f /dev/null
