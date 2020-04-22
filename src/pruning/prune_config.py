"""
@author: Stephen Baione (sbaione@email.sc.edu)
@description: Configuration for model pruning
"""


def get_supported_prune_levels():
    return "layer", "filter", "weight"


def get_supported_prune_methods():
    return "prune low magnitude", "taylor_first_order"
