import os
import socket
import json

from predict import predict_query
from logging_config import get_logger

logger = get_logger(__name__)

SOCKET_PATH = "/tmp/mysqld-ml.sock"

if os.path.exists(SOCKET_PATH):
    os.unlink(SOCKET_PATH)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0)
server.bind(SOCKET_PATH)
server.listen(1)

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
                    cardinality = predict_query(query_rep)
                    connection.sendall(str(cardinality).encode())
                data_buffer = messages[-1]