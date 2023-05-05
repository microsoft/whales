# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


def fb_score(precision, recall, b=1.0):
    """Computes the F_beta score.

    Args:
        precision: The precision.
        recall: The recall.
        b: The beta parameter.

    Returns:
        The F_beta score.
    """
    return (1 + b**2) * precision * recall / (b**2 * precision + recall)
