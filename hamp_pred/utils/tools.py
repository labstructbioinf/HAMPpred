def diffangle(targetA, sourceA):
    a = targetA - sourceA
    a = (a + 180) % 360 - 180
    return a
