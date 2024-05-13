import os
import logging

log_dir = 'log'
log_filename = "ml_log.log"

log_path = os.path.join(log_dir, log_filename)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def get_logger(name: str = None) -> logging.Logger:
    logging.basicConfig(
        filename=log_path,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a',  # Append mode
    )

    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)

    return logger