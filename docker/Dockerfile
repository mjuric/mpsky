FROM condaforge/miniforge3

# install mpsky
RUN mamba install -c conda-forge pytables tqdm pyarrow healpy pydantic-settings astropy pandas requests uvicorn fastapi
RUN git clone https://github.com/mjuric/mpsky.git
RUN cd mpsky && python -m pip install -e .
RUN mpsky --help

# FIXME: temporarily build a cache into the docker container
# In the future, the entrypoint will fetch the apropriate
# container from an AWS bucket
RUN mkdir /caches
COPY caches/* /caches/

# Usual entrypoint. Best to run with
# docker run -it --rm --tmpfs /tmp:size=100m -p 8000:8000 --read-only mpsky-daily --host 0.0.0.0 --verbose
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# switch to non-privileged user
USER ubuntu
RUN mpsky --help
