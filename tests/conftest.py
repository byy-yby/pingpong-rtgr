"""让 pytest 能 import src/ 下的 tabletennis 包。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
