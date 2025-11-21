#!/bin/bash
mkdir -p ../data ../db ../logs ../models
docker-compose -f ./docker/docker-compose.yml build

