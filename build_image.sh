#!/usr/bin/bash
#
# Copyright 2021.
# ozora-ogino

image_name=ozoraogino/spark-semseg:latest
docker build . -t $image_name
docker push $image_name
