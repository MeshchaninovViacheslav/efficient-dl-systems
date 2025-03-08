# grpc_service.py
import grpc
from concurrent import futures
import inference_pb2
import inference_pb2_grpc
from model import model  # from model.py

class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def Predict(self, request, context):
        url = request.url
        try:
            objects = model.predict(url)
            return inference_pb2.PredictReply(objects=objects)
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return inference_pb2.PredictReply()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    inference_pb2_grpc.add_InferenceServicer_to_server(InferenceServicer(), server)
    server.add_insecure_port('[::]:9090')
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
