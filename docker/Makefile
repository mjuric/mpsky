build:
	@ git diff --quiet HEAD -- . ':(exclude,top)docker/' || { git status; echo "*** detected uncommitted changes. Abort."; exit -1; }
	DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build . -t mpsky-daily --build-arg CACHE_BUST=$$(git describe) --build-arg MPSKY_BRANCH=$$(git branch --show-current)
	docker tag mpsky-daily mpsky-daily:$$(git branch --show-current)

TMPDIR ?= /tmp
run:
	docker run -it --rm -v $(TMPDIR):/tmp -p 8000:8000 --read-only mpsky-daily:$$(git branch --show-current) serve --host 0.0.0.0 --verbose

push:
	docker tag mpsky-daily ghcr.io/mjuric/mpsky-daily:$$(git branch --show-current)
	docker push ghcr.io/mjuric/mpsky-daily:$$(git branch --show-current)
