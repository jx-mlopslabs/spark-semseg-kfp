#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino

import argparse
import os
from io import BytesIO
from pathlib import Path
from pprint import pprint
from typing import Tuple
from urllib.parse import urlparse

import boto3
import numpy as np
import onnxruntime as rt
import pandas as pd
import wandb
from PIL import Image
from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StringType


def _to_image(image_schema_df: pd.DataFrame, index: int) -> Image:
    """Convert pandas DataFrame image schema to PIL.Image.
    Reference: https://github.com/apache/spark/blob/master/python/pyspark/ml/image.py#L143

    Args:
        image_schema_df(pandas.DataFrame): pandas version of Spark DataFrame Image schema.
        index(int): Index for DataFrame.

    Returns:
        image(PIL.Image):  Image with shape (height, width, channels)
    """
    byte_data, height, width, ch = (
        image_schema_df["data"][index],
        image_schema_df["height"][index],
        image_schema_df["width"][index],
        image_schema_df["nChannels"][index],
    )
    return Image.fromarray(
        np.asarray(list(byte_data), dtype=np.uint8).reshape([height, width, ch])
    )


def _upload_img_to_s3(
    img: Image,
    s3_bucket: str,
    s3_key: str,
):
    """Upload files to s3://<s3_bucket>/<s3_key>/predictions/<cityname>/

    Args:
        img(PIL.Image): Image data you want to upload
        s3_bucket(str): S3 bucket name.
        s3_key(str): S3 key.
        cityname(str): The city name of Cityscapes.
        file(str): Path of target file

    Raises:
        https://boto3.amazonaws.com/v1/documentation/api/latest/guide/error-handling.html#catching-exceptions-when-using-a-low-level-client
    """
    # Initialize S3 client with AWS secret.
    body = BytesIO()
    img.save(body, "PNG")
    body.seek(0)
    s3_client = boto3.client("s3")
    # Save image in s3://<bucket>/<key>/<cityname>/<image>.png
    res = s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=body)

    if res["ResponseMetadata"]["HTTPStatusCode"] == 200:
        print(f"Successfully uploaded: {Path(s3_bucket, s3_key)}")
    else:
        pprint(res)


def _blend(pred: np.ndarray, img: Image, alpha=0.5) -> np.ndarray:
    """Make alpha blending by prediction

    Args:
        preds(np.ndarray): Array which predicted by a Semseg model.
        img(PIL.Image): Original image with shape (height, width, channel).
        alpha(float): The interpolation alpha factor.

    Returns:
        blending(np.ndarray): Alpha blending result.
    """
    colormap = [
        (153, 153, 153),
        (250, 170, 30),
        (128, 64, 128),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
    ]

    colormap = np.array(colormap)
    colored = colormap[pred].astype("uint8")
    blending = Image.blend(
        Image.fromarray(colored[0, :, :, :]),
        img,
        alpha,
    )
    return blending


def _make_dest(origin: str) -> str:
    city, filename = origin.split("/")[-2:]
    dest = os.path.join(dest_path.value, "predictions", city, filename)
    return dest


@pandas_udf(StringType())
def inference_udf(image_df: pd.DataFrame) -> pd.Series:
    # def inference_udf(image_df: pd.DataFrame, dest_df: pd.Series) -> pd.Series:
    """Run ONNX inference and save all predictions on s3.

    Args:
        image_df(pd.DataFrame): Pandas DataFrame with columns of 'origin', 'height', 'width', 'nChannels', mode and 'data'.
                                For more detail, please follow the [Spark official document](https://spark.apache.org/docs/latest/ml-datasource.html).
        dest_df(pd.Series): Destination path.

    Returns:
        Currently dummy data is returned. You can return whatever you like.
        For example, evaluation results such as accuracy, MAE or etc.

    """
    # Download onnx model from W&B
    artifact_dir = wandb.PublicApi().artifact("mltools/sandbox/semseg:v0").download()
    sess = rt.InferenceSession(str(Path(artifact_dir, "model.onnx")))
    (_, _, width, height) = sess.get_inputs()[0].shape

    predictions = []

    for i in range(len(image_df)):
        # Convert Pandas dataframe to PIL.Image.
        img = _to_image(image_df, i).resize((height, width))
        # (height, width, channel) => (channel, height, width)
        img_array = np.transpose(np.array(img), axes=[2, 0, 1])

        # TODO: Remove here
        if img_array.shape[0] > 3:
            img_array = img_array[:3]
            img = Image.fromarray(np.array(img)[:, :, :3])

        pred = sess.run(
            ["output"],
            {"input": img_array[np.newaxis, ...].astype(np.float32)},
        )

        # Save the alpha blended result.
        blending = _blend(np.argmax(pred[0], axis=1), img)
        predictions.append(str(i))

        # If dest is on s3, upload predictions.
        dest = _make_dest(image_df["origin"][i])
        o = urlparse(dest)
        if o.scheme in ["s3a", "s3n"]:
            _upload_img_to_s3(img=blending, s3_bucket=o.netloc, s3_key=o.path[1:])

    return pd.Series(predictions)


def _setup_spark() -> Tuple[SparkContext, SparkSession]:
    conf = (
        SparkConf()
        .set(
            "spark.executor.extraJavaOptions",
            "-Dcom.amazonaws.services.s3.enableV4=true",
        )
        .set(
            "spark.driver.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true"
        )
    )
    context = SparkContext(conf=conf)
    session = SparkSession.builder.appName("cityscape-eval").getOrCreate()
    hadoopConf = context._jsc.hadoopConfiguration()

    context.setSystemProperty("com.amazonaws.services.s3.enableV4", "true")

    # Set S3 credentials in SparkContext
    AWS_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    hadoopConf.set("fs.s3n.awsAccessKeyId", AWS_ID)
    hadoopConf.set("fs.s3n.awsSecretAccessKey", AWS_KEY)
    hadoopConf.set("fs.s3a.endpoint", "s3.amazonaws.com")
    hadoopConf.set("com.amazonaws.services.s3a.enableV4", "true")
    hadoopConf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # This configuration enables efficient data loading from s3. (By default `noraml`)
    # In my case, this makes 10% faster than default.
    hadoopConf.set("fs.s3a.experimental.input.fadvise", "random")

    # Enable Apache Arrow. Arrow enables to use pandasUDF effectively.
    session.conf.set("spark.sql.execution.arrow.enabled", "true")
    session.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "64")
    return context, session


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()

    sc, spark = _setup_spark()
    dest_path = sc.broadcast(args.dest)

    # In this example, we use spark.read.format('image') which extracts meta data
    # and image bytearray as a Spark DataFrame Structure type.
    # For more information, refer the following.
    # https://spark.apache.org/docs/latest/ml-datasource.html#image-data-source
    df = (
        spark.read.format("image")
        .option("dropInvalid", True)
        .option("recursiveFileLookup", True)
        .load(
            args.src,
        )
        .select(inference_udf(col("image")).alias("prediction"))
    )

    df.write.mode("overwrite").save("./output")
    df.show()

    spark.stop()
