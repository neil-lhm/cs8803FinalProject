"""Stores common parameters for different model.
"""

def getTestSetRatio():
    """80/20 Train/Test split
    """
    return 0.2

def getColumnsToDrop():
    """Return columns to drop from training
    """
    return [
        'parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc',
        'propertycountylandusecode'
    ]
