import grpc
import inference_pb2
import inference_pb2_grpc

def run():
    # Connect to the gRPC server on localhost:9090
    with grpc.insecure_channel('localhost:9090') as channel:
        # Create a stub (client)
        stub = inference_pb2_grpc.InstanceDetectorStub(channel)
        
        # Create a request message
        request = inference_pb2.InstanceDetectorInput(url="http://images.cocodataset.org/val2017/000000001268.jpg")
        
        # Make the call to the Predict method
        response = stub.Predict(request)
        
        # Print the response
        print("Detected objects:", response.objects)

if __name__ == "__main__":
    run()