import grpc
from concurrent import futures
from typing import Callable, Optional

from dsbo.backends.utils.grpc import service_pb2, service_pb2_grpc


class RoDeOCallbackServicer(service_pb2_grpc.RoDeOCallbackServiceServicer):
    def __init__(self, 
                 objective_callback: Callable[[list], list],
                 constraint_callback: Optional[Callable[[list], list]] = None):
        self.objective_callback = objective_callback
        self.constraint_callback = constraint_callback
    
    def EvaluateObjective(self, request, context):
        design_vector = list(request.values)
        objective_values = self.objective_callback(design_vector)
        return service_pb2.ObjectiveValues(value=objective_values)
    
    def EvaluateGeometryConstraints(self, request, context):
        if self.constraint_callback is None:
            return service_pb2.ConstraintValues(values=[])
        
        design_vector = list(request.values)
        constraint_values = self.constraint_callback(design_vector)
        return service_pb2.ConstraintValues(values=constraint_values)


class GrpcServer:
    def __init__(self, 
                 objective_callback: Callable[[list], list],
                 constraint_callback: Optional[Callable[[list], list]] = None,
                 address: str = "localhost:50051",
                 max_workers: int = 10):
        
        self.address = address
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
        
        servicer = RoDeOCallbackServicer(objective_callback, constraint_callback)
        service_pb2_grpc.add_RoDeOCallbackServiceServicer_to_server(servicer, self.server)
        
        self.server.add_insecure_port(address)
    
    def start(self):
        self.server.start()
        print(f"Server started on {self.address}")
    
    def stop(self, grace: Optional[float] = None):
        self.server.stop(grace)
    
    def wait_for_termination(self):
        self.server.wait_for_termination()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


