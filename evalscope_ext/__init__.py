"""
evalscope_ext — stratified benchmark pruning extension for evalscope.

Install into your evalscope fork:
    pip install -e path/to/task2/

Then register datasets by importing this module at evalscope startup
(evalscope picks up entry_points in evalscope.benchmarks namespace).
"""
from evalscope_ext.datasets import register_all

register_all()
