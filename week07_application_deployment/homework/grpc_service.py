import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc
from model import model


class InferenceServicer(inference_pb2_grpc.InstanceDetectorServicer):
    def Predict(self, request, context):
        url = request.url
        try:
            objects = model.predict(url)
            return inference_pb2.InstanceDetectorOutput(objects=objects)
        except Exception as e:
            print(f"Error in Predict", flush=True)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return inference_pb2.InstanceDetectorOutput()


def serve():
    # Create a gRPC server with a thread pool of 4 workers to handle incoming requests concurrently.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    
    # Add the InstanceDetector service to the server.
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(InferenceServicer(), server)
    
    # Bind the server to listen on all network interfaces at port 9090 without using secure channels.
    server.add_insecure_port('[::]:9090')
    
    # Start the gRPC server, which begins processing incoming requests.
    server.start()
    
    # Block the thread until the server is terminated, ensuring the server keeps running.
    server.wait_for_termination()


if __name__ == "__main__":
    serve()