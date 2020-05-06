"""
@author: Stephen Baione (sbaione@email.sc.edu)
@description: Configuration for model pruning
"""


def get_supported_prune_levels():
    return {"taylor_first_order": "filter", "prune_low_magnitude": "weight"}


def get_supported_prune_methods():
    return "prune low magnitude", "taylor_first_order"
