FROM condaforge/miniforge3

ARG MPSKY_BRANCH="main"

RUN mamba install -c conda-forge pytables tqdm pyarrow healpy pydantic-settings astropy pandas requests uvicorn fastapi
RUN apt-get update && apt-get install -y bc curl && apt-get clean && rm -rf /var/lib/apt/lists/*

# install mpsky from source
ARG CACHE_BUST
RUN echo "Forcing rebuild at ${CACHE_BUST}, branch=${MPSKY_BRANCH}"
RUN git clone https://github.com/mjuric/mpsky.git
RUN cd mpsky && git checkout ${MPSKY_BRANCH} && python -m pip install -e .
RUN mpsky --help

# Usual entrypoint. Best to run with
# docker run -it --rm --tmpfs /tmp:size=100m -p 8000:8000 --read-only mpsky-daily --host 0.0.0.0 --verbose
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# switch to non-privileged user
USER ubuntu
ENV MPLCONFIGDIR=/tmp
