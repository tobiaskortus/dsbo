import os
import grpc
import pandas as pd
from typing import List

from dsbo.backends.utils.grpc import service_pb2, service_pb2_grpc

class GRPCClient:
    def __init__(self, address: str = "localhost:50051"):
        self.address = address
        self.channel = grpc.insecure_channel(address)
        self.stub = service_pb2_grpc.RoDeOCallbackServiceStub(self.channel)
    
    def evaluate_objective(self, design_vector: List[float]) -> List[float]:
        request = service_pb2.DesignVector(values=design_vector)
        response = self.stub.EvaluateObjective(request)
        return list(response.value)
    
    def evaluate_constraints(self, design_vector: List[float]) -> List[float]:
        request = service_pb2.DesignVector(values=design_vector)
        response = self.stub.EvaluateGeometryConstraints(request)
        return list(response.values)
    
    def close(self):
        self.channel.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from argparse import ArgumentParser

    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    argparse = ArgumentParser()
    argparse.add_argument("-w", "--workdir", type=str, default="/scratch/nyl93mev/dsbo/files/run_rodeo_0/rodeo") #required=True
    #argparse.add_argument("-q", "--query", type=str, required=True)
    #argparse.add_argument("-a", "--address", type=str, default="localhost:50051")

    args = argparse.parse_args()

    with open(os.path.join(args.workdir, 'dv.dat'), "r") as f:
        design = list(map(float, f.readlines()))      

    with GRPCClient("localhost:50051") as client:
        objective = client.evaluate_objective(design)

        with open(os.path.join(args.workdir, 'objective.dat'), "w") as f:
            f.write(f"{float(objective[0])}")
        
