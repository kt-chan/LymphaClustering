#!/bin/bash
cd /home/demo/projects/ruipath/LymphaClustering
cp -f ./.env ./.env-bak
cp -rf ./.env-example ./.env
rm -rf ./db ./logs
mkdir -p ./data ./db ./logs
docker-compose -f ./docker/docker-compose-win.yml build

