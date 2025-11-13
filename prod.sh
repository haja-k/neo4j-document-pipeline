#!/bin/bash
docker compose -f docker-compose.prod.yml up -d --scale api=2 --build