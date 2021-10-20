#!/usr/bin/env python3
#
# Copyright 2021.
# ozora-ogino
# pylint: disable=pointless-string-statement,trailing-whitespace,unused-argument,redefined-outer-name,reimported

import argparse
from pathlib import Path

import kfp
import kfp.compiler as compiler
import kfp.dsl as dsl
import yaml
from kfp.components import func_to_container_op
from kfp.onprem import use_k8s_secret

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cityscapes")
parser.add_argument("--type", default="boto3")
args = parser.parse_args()

resource = yaml.safe_load(Path(f"manifests/{args.dataset}/{args.type}.yaml").read_text())


def _verify_completion(name: str, min_interval: int = 3) -> None:
    """
    This function executes:
    1. Verifing whether Spark application has been completed.
    2. Logging executor's status.
    3. Printing Driver's log if the Spark application is failed.

    Args:
        name(str): Spark application's name.
        min_interval: Time inteval (minutes).
    """
    import datetime
    import subprocess

    from retrying import retry

    @retry(
        wait_fixed=min_interval * 60000,
        retry_on_exception=lambda _: False,
        retry_on_result=lambda ret: ret != "COMPLETED",
    )
    def _check_completion(name: str) -> str:
        """
        Check whther a Spark application has been completed
        by checking `sparkapplication.status.applicationState.state`.
        If the state is Failed, print driver's log before it is deleted.
        For the detail of Spark application's state, see the link below.
        https://github.com/GoogleCloudPlatform/spark-on-k8s-operator/blob/83312b9c4754d57175f4d698638cf3668395dbd8/pkg/controller/sparkapplication/controller.go#L457

        Args:
            name(str): Spark application's name.

        Returns:
            state(str): Spark application's state.

        Raises:
            subprocess.CalledProcessError: If Spark application is deleted, raise an exception.
            RuntimeError: If Spark application is failed, raise an exception.

        """
        print(f"\n---------------- {datetime.datetime.now()} ----------------")

        # Print executor's status
        subprocess.run(
            f"""
                set -e;
                tmp=`kubectl get sparkapplications.sparkoperator.k8s.io {name} -o json`;
                echo ${{tmp}} | jq .status.executorState;
            """,
            shell=True,
            check=True,
        )

        # Get SparkApplication's status.
        state = (
            subprocess.run(
                f"kubectl get sparkapplication {name} -o json | jq '.status.applicationState.state'",
                shell=True,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            # e.g. "RUNNING\n" -> RUNNING
            .stdout.lstrip('"').rstrip('\n"')
        )

        # If SparkApplication is failed, print driver's log and raise RuntimeError.
        if state == "FAILED":
            print("\n---------------- Driver log ----------------\n")
            subprocess.run(f"kubectl logs {name}-driver", shell=True, check=True)
            raise RuntimeError("Spark Application has been Failed somehow.")

        return state

    # Install kubectl and jq.
    # For avoiding too many Docker image, install dependencies by `subprocess`.
    subprocess.run(
        """
        curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" &&
        install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl &&
        apt-get update &&
        apt-get install -y jq
        """,
        stdout=subprocess.DEVNULL,
        shell=True,
        check=True,
    )
    _check_completion(name)


def _verify_job(src_s3_uri: str, dest_s3_uri: str):
    """
    Verify whether Spark application has been completed
    by comparing the number of images in the given s3 folders.
    Args:
        src_s3_uri(str): s3a URI like "s3a://{bucket_name}/{cityscapes_prefix}/leftImg8bit/test".
        dest_s3_uri(str): s3a URI like "s3a://{your_bucket}/{your_prefix}".
    Raises:
        If the number of images are not same between two s3 folder, raises an exception.
    """
    from pathlib import Path

    import boto3

    def _count_images(s3_uri: str) -> int:
        """
        Count images under `s3_uri`.
        """
        ret = 0
        bucket_name, prefix = s3_uri[len("s3a://") :].split("/", 1)
        bucket = boto3.resource("s3").Bucket(bucket_name)
        for o in bucket.objects.filter(Prefix=prefix):
            if Path(o.key).suffix in [".png", ".jpg", ".jpeg"]:
                ret += 1
        print(f"{ret} images are found under {s3_uri}")
        return ret

    if _count_images(src_s3_uri) != _count_images(dest_s3_uri):
        raise RuntimeError("Job failed for some reason")
    print("Job succeeded.")


# %%
verify_completion_op = func_to_container_op(
    _verify_completion, base_image="python:3.8", packages_to_install=["retrying"]
)

verify_job_op = func_to_container_op(_verify_job, packages_to_install=["boto3"])


@dsl.pipeline(
    name="Cityscapes semseg evaluation with pyspark",
    description="",
)
def pipeline():

    aws_secret = use_k8s_secret(
        secret_name="aws-auth",
        k8s_secret_key_to_env={
            "aws_access_key_id": "AWS_ACCESS_KEY_ID",
            "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        },
    )

    spark_task = dsl.ResourceOp(
        name="Spark evaluation",
        k8s_resource=resource,
        action="create",
        success_condition="status.applicationState.state == RUNNING",
        failure_condition="status.applicationState.state == FAILED",
    ).add_pod_annotation("pipelines.kubeflow.org/max_cache_staleness", "P0D")
    # Disabling cache.

    # Verifing the inference process.
    verify_state_task = verify_completion_op(resource["metadata"]["name"], min_interval=1).after(spark_task)
    src_s3_arg, dest_s3_arg = resource["spec"]["arguments"]
    src_s3_uri, dest_s3_uri = src_s3_arg.split("=")[-1], dest_s3_arg.split("=")[-1]
    verify_job_op(src_s3_uri, dest_s3_uri).apply(aws_secret).after(verify_state_task)


EXPERIMENT_NAME = "sandbox"
namespace = "kubeflow"
pipeline_filename = "pipeline.yaml"

# Cookie from browser.
cookie = "YOUR_COOKIE"
authservice_session = "authservice_session=" + cookie

compiler.Compiler().compile(pipeline, pipeline_filename)
client = kfp.Client(namespace=namespace)
experiment = client.create_experiment(EXPERIMENT_NAME, namespace=namespace)
run_name = "spark inference"
run_result = client.run_pipeline(experiment.id, run_name, pipeline_filename)

# %%
