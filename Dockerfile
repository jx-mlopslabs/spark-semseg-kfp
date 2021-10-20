# Copyright 2021.
# ozora-ogino

FROM gcr.io/spark-operator/spark-py:v3.1.1

COPY src/* /opt/src/

USER root
RUN apt-get install -y wget
RUN pip3 install --no-cache-dir pandas onnxruntime pillow wandb pyarrow boto3

WORKDIR /src/

RUN wget -P /opt/spark/jars https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/2.7.4/hadoop-aws-2.7.4.jar
RUN wget -P /opt/spark/jars https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.7.4/aws-java-sdk-1.7.4.jar