from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Write all the libary version that is mention

print(f"joblib version: {joblib.__version__}")
print(f"pickle version: {pickle.version}")
print(f"datetime version: {date.__module__}")
print(f"pathlib version: {Path.__module__}")
print(f"timedelta version: {timedelta.__module__}")
print(f"annotation version: {annotations.__module__}")