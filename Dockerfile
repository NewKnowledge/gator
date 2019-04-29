# start from a pinned version of the miniconda image which provides python 3 on debian
FROM continuumio/miniconda3:4.5.12

# /root is the home directory for the root user, all commands below are run relative to /root with root user priviledges
ENV HOME=/root
WORKDIR $HOME

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# update os package manager, then install prerequisite packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git
# build-essential

# update conda, then install required python packages in environment.<group>.yml
COPY environment.yml $HOME/
RUN conda update -n base conda && \
    conda env update -f $HOME/environment.yml

# a place to install new requirements while developing without rebuilding the core packages
# COPY environment.tmp.yml $HOME/
# RUN conda env update -f $HOME/environment.tmp.yml

# copy everything else (excluding stuff specified in .dockerignore)
COPY . $HOME/

# pip install this package so that it is accessible anywhere (e.g. in a notebook or web api)
RUN pip install -e .

# by default, run the main script
CMD ["python", "scripts/main.py"]
