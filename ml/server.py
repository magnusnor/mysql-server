import os
import socket
import json

from predict import predict_query
from logging_config import get_logger

# def uint64_to_bitmap(value):
#     # Create a bitmap list with 64 bits (0 or 1) representing the value
#     return [(value >> i) & 1 for i in reversed(range(64))]

# def display_bitmap(bitmap):
#     print('[' + ', '.join(map(str, bitmap)) + ']')

logger = get_logger(__name__)

SOCKET_PATH = "/tmp/mysqld-ml.sock"

if os.path.exists(SOCKET_PATH):
    os.unlink(SOCKET_PATH)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0)
server.bind(SOCKET_PATH)
server.listen(1)

# while True:

#     logger.info('Server is listening for incoming connections...')
#     connection, client_address = server.accept()
#     logger.info(f'Connection from: {str(connection.getsockname())}')

    # while True:
    #     data = connection.recv(2048)
    #     if not data:
    #         break
    #     decoded_data = data.decode()
    #     logger.info(f"Decoded data: {decoded_data}, Type: {type(decoded_data)}")

    #     # bitmap = uint64_to_bitmap(int(float(decoded_data)))
    #     # logging.info(f"Bitmap: {bitmap}")
    #     # connection.sendall(str(bitmap).encode())
    #     # display_bitmap(bitmap)

    #     pattern = r'\(\s*(.*?)\s*=\s*(.*?)\s*\)'
    #     result = re.sub(pattern, r'\1=\2', decoded_data)

    #     query = dict()
    #     query["tables"] = [result]
    #     query["joins"] = [result]
    #     query["predicates"] = [result]

    #     logger.info(query)
        
    #     cardinality = predict_query(query)
    #     connection.sendall(str(cardinality).encode())

while True:
    logger.info('Server is listening for incoming connections...')
    connection, _ = server.accept()
    logger.info(f'Connection from: {str(connection.getsockname())}')
    with connection:
        data_buffer = ""
        while True:
            data = connection.recv(1024)
            if not data:
                break
            data_buffer += data.decode()
            # Process messages split by the newline delimiter
            if "\n" in data_buffer:
                messages = data_buffer.split('\n')
                for msg in messages[:-1]:
                    query_rep = json.loads(msg)
                    logger.info(f"Received query representation from MySQL: {query_rep}")
                    # query = {
                    #     'tables': [query_rep["tables"]],
                    #     'joins': [query_rep["joins"]],
                    #     'predicates': [query_rep["predicates"]]
                    # }
                    # logger.info(f"Query representation: {query}")
                    cardinality = predict_query(query_rep)
                    connection.sendall(str(cardinality).encode())
                data_buffer = messages[-1]