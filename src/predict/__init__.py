import sys

from utils.pathtools import project

if project.root not in sys.path:
    sys.path.append(project.root)