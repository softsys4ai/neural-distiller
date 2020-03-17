"""
@author: Stephen Baione (sbaione@email.sc.edu)
@description: Configuration for model pruning
"""


def get_supported_prune_levels():
    return "layer", "filter", "weight"


def get_supported_prune_methods():
    return "oracle", "taylor_first_order", "taylor_second_order"
