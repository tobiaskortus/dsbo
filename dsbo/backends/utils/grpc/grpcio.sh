#!/bin/bash
cd dsbo/backends/utils/grpc
python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. service.proto
sed -i 's/^import service_pb2 as service__pb2/import dsbo.backends.utils.grpc.service_pb2 as service__pb2/' service_pb2_grpc.py