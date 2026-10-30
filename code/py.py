"""项目公共依赖。

其余模块通过 ``from py import *`` 使用这里统一导入的依赖，避免在各文件中
重复维护第三方库引用。
"""

from brokenaxes import brokenaxes
import ast
import csv
from collections import Counter
from collections import defaultdict
import copy
from datetime import datetime
from datetime import timedelta
from functools import lru_cache
from gc import collect
import gzip
import hashlib
import hypernetx as hnx
import io
from itertools import chain
from itertools import combinations
from itertools import permutations
from itertools import product
import json
import math
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import ticker
from matplotlib import cm
import networkx as nx
import numba
import numpy as np
import os
import pandas as pd
import pickle as pkl
import re
import requests
import seaborn as sns
import sklearn.cluster
from sklearn import metrics
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp
from scipy import stats
from scipy.special import comb
from scipy.special import gammaln
from scipy.special import logsumexp
import sys
import threading
import time
import tempfile
from tqdm.auto import tqdm
import warnings
import xgi
import zlib

try:
    import hypergraphx as hgx
except ImportError:
    hgx = None

try:
    import truststore
except ImportError:
    truststore = None
