import alc

TOL = 1e-8

def esPseudoInversa(A, B, tol=TOL):
    AB = alc.matMul(A, B)
    BA = alc.matMul(B, A)
    cond1 = alc.matricesIguales(alc.matMul(AB, A), A, tol=tol)
    cond2 = alc.matricesIguales(alc.matMul(BA, B), B, tol=tol)
    cond3 = alc.matricesIguales(alc.transpuesta(AB), AB, tol=tol)
    cond4 = alc.matricesIguales(alc.transpuesta(BA), BA, tol=tol)
    return cond1 and cond2 and cond3 and cond4