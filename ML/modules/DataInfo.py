"""
Some constants about the data
"""

luminosity = 100

signalFullSize = [50000, 50000, 49600, 50000, 50000, 50000, 100000, 100000, 100000, 99800]
signalSkimSize = [403, 394, 390, 372, 479, 655, 1619, 1858, 2008, 2239]
signalXSec = [1.0005075368, 0.9138817159, 0.8375223919, 0.7075961742, 0.6042382072,
              0.3130310897, 0.1715597927, 0.1335982543, 0.1057026495, 0.0681296695]
signalAcceptance = [a/b for a,b in zip(signalSkimSize, signalFullSize)]

#allSignalFullSize = sum(signalFullSize)
#allSignalSkimSize = sum(signalSkimSize)
#allSignalAcceptance = allSignalSkimSize/allSignalFullSize

backgroundFullSize = [92925926, 47502020]
backgroundSkimSize = [166157, 1896]
backgroundXSec = [816, 61526.7]
backgroundAcceptance = [a/b for a,b in zip(backgroundSkimSize, backgroundFullSize)]

#allBackgroundFullSize = sum(backgroundFullSize)
#allBackgroundSkimSize = sum(backgroundSkimSize)
#allBackGroundAcceptance = allBackgroundSkimSize/allBackgroundFullSize