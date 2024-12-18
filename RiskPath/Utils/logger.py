########################################################################################################################
# Apache License 2.0
########################################################################################################################
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2024 Nina de Lacy

########################################################################################################################
# Import packages
########################################################################################################################
import atexit
import logging
import os
import sys
from datetime import datetime
from typing import Optional

########################################################################################################################
# Define a helper class to encapsulate the logging functionality
########################################################################################################################

class ConsoleToLogger:
    # A helper class to
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():                              # Ignore empty messages (like newlines)
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

########################################################################################################################
# Define the main function for the logging functionality
########################################################################################################################

def create_log(filename: Optional[str] = None):
    """
    :param filename: A string or None.
           Name of the returned logging file. Date and time in '%Y_%m_%d_%H_%M_%S' format if None.
           Default setting: filename=None.
    :return:
    A logging file (in .log format) created in the current working directory.
    """
    start_time = datetime.now()

    # Type and value check
    if filename is not None:
        assert isinstance(filename, str), \
            f'filename (if not None) must be a string. Now its type is {type(filename)}.'
    else:
        filename = start_time.strftime('%Y_%m_%d_%H_%M_%S')

    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=[logging.FileHandler(f"{filename}.log"), logging.StreamHandler()])
    sys.stdout = ConsoleToLogger(logging.getLogger(), logging.INFO)
    sys.stderr = ConsoleToLogger(logging.getLogger(), logging.ERROR)
    print('='*100, flush=True)
    print(f'Logging of {os.path.basename(sys.argv[0])}', flush=True)
    print(f"Start time: {start_time.strftime('%Y_%m_%d_%H_%M_%S')}", flush=True)
    print('Logging starts.', flush=True)
    print('='*100, flush=True)

    def on_exit():
        print('='*100, flush=True)
        print(f"Logging ends.", flush=True)
        end_time = datetime.now()
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"Elapsed time: {end_time - start_time}", flush=True)
        print('='*100, flush=True)

    atexit.register(on_exit)

########################################################################################################################
