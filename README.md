## Semantic Segmentation with Spark on k8s

Semantic segmentation inference with Spark.

This project uses stuffs below.
- Spark
- Kubeflow

    To emphasize debuggability of Spark operator, this project adopts Kubeflow.
	In `pipeline.py`, useful functions for debugging and logging is implemented.

- ONNX

    This project assume that trained model is compiled to ONNX format.

- S3 (dataset store)
- W&B (model store)

	This project assume model is stored in W&B artifact as ONNX format.

## Usage

```bash
python pipeline.py --dataset <DATASET> --type <METHOD_TYPE>
```

- DATASET: cityscapes or imagenet.
- METHOD_TYPE: boto3 or s3a.
