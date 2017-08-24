from math import sqrt
from DataInfo import *


def signalSignificanceSelector(signalSignificance):
    temp = sorted(signalSignificance)[0:min(len(signalSignificance), 3)]
    return sum(temp)/len(temp)

def calculateFOM(len_signals, len_backgounds, signalIdx, backgroundIdx, FOMselector=None):
    s_signal, s_background = [], 0
    for i, signal in enumerate(len_signals):
        s_signal.append(signal * signalXSec[signalIdx[i]] / signalFullSize[signalIdx[i]] * luminosity)

    for i, background in enumerate(len_backgounds):
        s_background += background * backgroundXSec[backgroundIdx[i]] / backgroundFullSize[backgroundIdx[i]] * luminosity

    if s_background != 0 and not FOMselector:
        return signalSignificanceSelector(s_signal) / sqrt(s_background)
    elif s_background != 0:
        return FOMselector(s_signal) / sqrt(s_background)
    else:
        return -1

def mergeDataFrame(data):
    """
    merge list of pandas.DataFrame ignoring index
    """
    allData = data[0]
    for _data in data[1:]:
        allData = allData.append(_data, ignore_index=True)
    return allData