import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        self.input_data_path = None
        self.output_path = None

    def set_input_data_path(self, path):
        self.input_data_path = path

    def set_output_path(self, path):
        self.output_path = path

class Constants:
    LOG_LEVEL = logging.INFO