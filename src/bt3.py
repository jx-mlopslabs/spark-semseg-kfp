#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino

import argparse
import os
from io import BytesIO
from pathlib import Path
from typing import List, Tuple
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


def _fetch_img(origin: str) -> Image:
    """Fetch an image from S3 by boto3.

    Args:
        origin(str): S3 path like `<Bucket>/<Key>`.
                     (e.g. my_bucket/path/to/image.png)
    Returns:
        Image: An image loaded as a PIL.Image.
    """
    bucket, key = origin.split("/", 1)
    s3 = boto3.resource("s3").Bucket(bucket)
    obj = s3.Object(key)
    # Sending HTTP GET request via boto3.
    # https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html
    response = obj.get()
    file_stream = response["Body"]
    img = Image.open(file_stream)
    return img


def _upload_img(
    img: Image,
    s3_bucket: str,
    s3_key: str,
):
    """Upload files to s3://<s3_bucket>/<s3_key>

    Args:
        img(PIL.Image): Image data you want to upload.
        s3_bucket(str): S3 bucket name.
        s3_key(str): S3 key.
    """
    # Initialize S3 client with AWS secret.
    body = BytesIO()
    img.save(body, "PNG")
    body.seek(0)
    s3_client = boto3.client("s3")
    # Save image in s3://<bucket>/<prefix>/<cityname>/<image_name>.png
    s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=body)


def _class2color(class_map: np.ndarray) -> np.ndarray:
    """
    Colorize a 2D map of class IDs according to a hard-coded color map.
    Args:
        class_map (np.ndarray): 2D array holding class IDs.
    Returns:
        np.ndarray: RGB array in the HWC format.
    """
    rgbs = [
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
    return np.array(rgbs, dtype=np.uint8)[class_map]


def _make_dest(origin: str) -> str:
    """Create destination path from origin path.
    Args:
        origin(str): S3 path like `<Bucket>/<Key>`
    Returns:
        str: Destination path.
    """
    # my_bucket/cityscapes/leftImg8bit/test/cityname/img.png
    # => cityname, img.png
    city, filename = origin.split("/")[-2:]
    dest = os.path.join(DEST_PATH, "predictions", city, filename)
    return dest


@pandas_udf(StringType())
def _inference_udf(origin_df: pd.DataFrame) -> pd.Series:
    """Run ONNX inference and save all predictions on s3.

    Args:
        image_df(pd.DataFrame): Pandas DataFrame with columns of 'origin', 'height', 'width', 'nChannels', mode and 'data'.
                                For more detail, please follow the [Spark official document](https://spark.apache.org/docs/latest/ml-datasource.html).

    Returns:
        pd.Series: Currently dummy data is returned. You can return whatever you like.
        For example, evaluation results such as accuracy, MAE or etc.

    """
    # Set up ONNX Runtime with a W&B artifact.
    artifact_dir = wandb.PublicApi().artifact("mltools/sandbox/semseg:v0").download()
    sess = rt.InferenceSession(str(Path(artifact_dir, "model.onnx")))
    inputs = sess.get_inputs()
    wh = tuple(reversed(inputs[0].shape[len("bc") :]))
    outputs = sess.get_outputs()

    predictions = []

    for i in range(len(origin_df)):
        rgb = _fetch_img(origin_df[i]).convert("RGB").resize(wh)
        hwc = np.array(rgb).astype(np.float32)
        chw = np.transpose(hwc, axes=[2, 0, 1])
        bchw = np.expand_dims(chw, axis=0)

        # Inference.
        prob_maps = sess.run([outputs[0].name], {inputs[0].name: bchw})
        class_map = np.argmax(np.squeeze(prob_maps), axis=0)

        # Save the alpha-blended result.
        colorized_preds = _class2color(class_map)
        blended = Image.blend(Image.fromarray(colorized_preds), rgb, alpha=0.5)

        # If dest is on s3, upload predictions.
        dest = _make_dest(origin_df[i])
        o = urlparse(dest)
        if o.scheme in ["s3a", "s3n"]:
            _upload_img(img=blended, s3_bucket=o.netloc, s3_key=o.path[len("/") :])
        else:
            # Saving results in HDFS or FSX will be implemented in a future.
            pass

        # Spark assumes to save someting as a file with `spark.write`
        # In this sample, we use dummy results.
        # Of course you can replace this by whatever you like. (e.g.IoU)
        predictions.append(str(i))

    return pd.Series(predictions)


def _get_img_lists(src: str) -> List[str]:
    """Lists images contained in `src`.
    Args:
        src(str): S3 path like `<Bucket>/<Prefix>`.
                  (`e.g. your_bucket/datasets/cityscapes/leftImg8bit/test/)
    Returns:
        List[str]: List of images contained in `src`
    """
    o = urlparse(src)
    bucket, prefix = o.netloc, o.path[len("/") :]
    s3 = boto3.resource("s3").Bucket(bucket)

    def _is_image(filename: str) -> bool:
        exts = ["png", "jpg", "jpeg", "PNG", "JPEG", "JPG"]
        ext = filename.split(".")[-1]
        return ext in exts

    files = [
        os.path.join(bucket, x.key)
        for x in s3.objects.filter(Prefix=str(prefix))
        if _is_image(x.key)
    ]
    return files


def _setup_spark() -> Tuple[SparkContext, SparkSession]:
    """Configure Spark Context and Spark Session."""
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
    session = SparkSession.builder.appName("cityscape").getOrCreate()
    hadoopConf = context._jsc.hadoopConfiguration()

    context.setSystemProperty("com.amazonaws.services.s3.enableV4", "true")

    # Set S3 credentials in SparkContext.
    AWS_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    hadoopConf.set("fs.s3n.awsAccessKeyId", AWS_ID)
    hadoopConf.set("fs.s3n.awsSecretAccessKey", AWS_KEY)
    hadoopConf.set("fs.s3a.endpoint", "s3.amazonaws.com")
    hadoopConf.set("com.amazonaws.services.s3a.enableV4", "true")
    hadoopConf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

    # Enable Apache Arrow. Arrow enables to use pandasUDF effectively.
    session.conf.set("spark.sql.execution.arrow.enabled", "true")
    session.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "64")
    return context, session


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dest", required=True)
    args = parser.parse_args()

    # We use dest path as a global variable for pandasUDF.
    DEST_PATH = args.dest

    sc, spark = _setup_spark()

    # 1. Create Spark DataFrame by the list of S3 path.
    # 2. Run inference as a pandasUDF (User Defined Function).
    origins = _get_img_lists(args.src)
    df = spark.createDataFrame(pd.DataFrame({"origin": origins})).select(
        _inference_udf(col("origin")).alias("prediction")
    )

    # To make sure that Spark Application will be completed, you must consume whole DataFrame.
    # e.g. Calling `write.save(...)` or `df.show(df.count(), False)`.
    # In this sample, dummy results are written to `dest`.
    df.write.mode("overwrite").save(os.path.join(DEST_PATH, "output"))

    spark.stop()
