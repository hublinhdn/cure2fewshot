"""Test fixtures. The protocol scripts are plain modules under protocol/; put that directory on
sys.path so tests import them by name, and keep every test independent of the collection: no
test reads an image of the collection or an embedding of it."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "protocol"))
