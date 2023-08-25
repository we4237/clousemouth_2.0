import os
import sys

_current_python_file = os.path.abspath(os.path.dirname(__file__))
if sys.path.count(_current_python_file) == 0:
    sys.path.insert(0, _current_python_file)
