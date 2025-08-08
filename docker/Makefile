build:
	DOCKER_DEFAULT_PLATFORM=linux/amd64 docker build . -t mpsky-daily

push:
	docker tag mpsky-daily ghcr.io/mjuric/mpsky-daily:latest
	docker push  ghcr.io/mjuric/mpsky-daily:latest
