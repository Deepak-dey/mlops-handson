.PHONY: check up ps down logs send results predict checks

STAMP := $(shell date +%Y%m%d-%H%M)

check:   ## build and start as duration-service:<date>-<time>-<n>
	docker compose config --quiet

up:
	@n=1; while image inspect duration-service:$(STAMP)-$$n >/dev/null 2>&1; do n=$$((n+1)); done; \
	tag=$(STAMP)-$$n; IMAGE_TAG=$$tag docker compose up -d --build && \
	docker tag duration-service:$$tag duration-service:latest && echo "runnung duration-service:$$tag"

ps:    ## containers and their health
	docker compose ps

down:
	docker compose down

logs:
	docker compose logs -f service

send: ## put one ride on the `rides` topic
	echo '{"ride_id":"r1","PULocationID":43,"DOLocationID":151,"trip_distance":1.2}' | docker compose exec -T redpanda rpk topic produce rides

results: ## read what the service wrote to the `predictions` topic
	docker compose exec redpanda rpk topic consume predictions -o start -n 1

predict: ## one ride through the HTTP API
	curl -s -X POST localhost:9096/predict -H 'Content-Type: application/json' -d '{"PULocationID": 43, "DOLocationID": 151, "trip_distance": 1.2}'

checks:          ## up + ps + black + isort + pylint + pytest   (FIX=1 make checks lets black/isort rewrite)
	./test/run_checks.sh


