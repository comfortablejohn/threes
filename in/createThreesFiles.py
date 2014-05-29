"""
This program creates input files for the 2014 CITS3001 project.
It creates a file filename with roughly noOfPieces tiles whose
relative proportions are specified by the list proportions.
This version sets up a standard initial board. 

v1.1, Lyndon While, 24/4/14 
"""


import random


#this is the output file name
filename = "ThreesInput.txt"
#this is the total number of pieces
noOfPieces = 30
#these are the proportions of 1, 2, 3, 6, 12, etc.
#the proportions are summed and rounded, so the number of pieces returned won't always be exact
#the easiest way to make it exact is for proportions to contain integers that sum to noOfPieces
proportions = [12, 12, 3, 1, 1, 1]


def normaliseProportions():
#normaliseProportions() replaces the values on proportions with the corresponding numbers of pieces
    z = sum(proportions)
    for k in range(len(proportions)):
        proportions[k] = round(proportions[k] * noOfPieces / z)


def randomisePieces():
#randomPieces() randomises the items on pieces
#it swaps each piece with a randomly-chosen piece further along the list
    l = len(pieces) - 1
    for k in range(l):
        z         = random.randint(k + 1, l)
        t         = pieces[k]
        pieces[k] = pieces[z]
        pieces[z] = t


def piece(k):
#piece(k) returns the kth smallest tile 
    if k <= 1:
        return k + 1
    else:
        return 3 * 2 ** (k - 2)


if any([x < 0 for x in proportions]):
    print("Proportions must contain only non-negative numbers")
elif sum(proportions) <= 0:
    print("Proportions must contain at least one positive number")
else:
    normaliseProportions()
    #print(proportions)
    pieces = [x for k in range(len(proportions)) for x in [piece(k)] * int(proportions[k])]
    randomisePieces()
    #print(pieces)
    maxPiecesPerLine = 20
    ps = [pieces[k:k + maxPiecesPerLine] for k in range(0, len(pieces), maxPiecesPerLine)]
    f = open(filename, 'w')
    f.write("\n".join([str(len(pieces)) + " pieces; " + str(proportions)[1:-1],
            "", "0 0 0 0", "0 1 2 0", "0 2 1 0", "0 0 0 0", ""] + [" ".join([str(x) for x in l]) for l in ps]))
    f.close()
