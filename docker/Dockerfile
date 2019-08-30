# Start from anaconda image
FROM continuumio/anaconda:latest

# Install python dev and latex libraries
RUN apt update
RUN apt install -y python-dev texlive-fonts-recommended texlive-fonts-extra dvipng

# Set display environment variable
ENV QT_X11_NO_MITSHM=1

# Set up home directory in image (this is where the repo directory will be mounted)
RUN mkdir /home/LMPC
WORKDIR /home/LMPC

# Install jupyter and python-tk (backend for matplotlib.pyplot)
RUN /opt/conda/bin/conda install -c anaconda jupyter tk -y
# Install cvxpy and ipopt
RUN /opt/conda/bin/conda install -c conda-forge cyipopt cvxpy -y

# Delete matplotlib font cache so we can rebuild it later
RUN fc-cache -f -v
RUN rm -rf /root/.cache/matplotlib

# Two options for command to run

# Start jupyter notebook server
CMD /opt/conda/bin/jupyter notebook --notebook-dir=/home/LMPC --ip=0.0.0.0 --port=8888 --allow-root

# Keep container open after creation
#CMD tail -f /dev/null
