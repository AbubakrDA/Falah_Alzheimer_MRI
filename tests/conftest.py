import os
import sys
import ctypes
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Set in Python's os.environ
os.environ["USERPROFILE"] = str(ROOT_DIR)
os.environ["HOME"] = str(ROOT_DIR)
os.environ["KERAS_HOME"] = str(ROOT_DIR / ".keras")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set in OS process environment table via Win32 API
try:
    kernel32 = ctypes.windll.kernel32
    kernel32.SetEnvironmentVariableW("USERPROFILE", str(ROOT_DIR))
    kernel32.SetEnvironmentVariableW("HOME", str(ROOT_DIR))
    kernel32.SetEnvironmentVariableW("KERAS_HOME", str(ROOT_DIR / ".keras"))
    kernel32.SetEnvironmentVariableW("TF_ENABLE_ONEDNN_OPTS", "0")
    kernel32.SetEnvironmentVariableW("TF_CPP_MIN_LOG_LEVEL", "2")
except Exception:
    pass

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
