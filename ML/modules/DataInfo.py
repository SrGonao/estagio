"""
Some constants about the data
"""

luminosity = 100

signalFullSize = [50000, 50000, 49600, 50000, 50000, 50000, 100000, 100000, 100000, 99800]
signalSkimSize = [403, 394, 390, 372, 479, 655, 1619, 1858, 2008, 2239]
signalXSec = [2.4524985139, 3.2882620826, 3.8318006272, 4.5185294336, 4.9327584051,
              5.6041453978, 5.7246882202, 5.6830461542, 5.5990314598, 5.3528318771]
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