#!/bin/bash
docker build -t wq-model-serve-i .
docker run -d --name wq-model-serve --network=wq-admin-be_line-connector --restart=always wq-model-serve-i 