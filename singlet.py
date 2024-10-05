import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

lambdaH = (125 * 125) / (246 * 246)
v = 246
nh = 1
nt = -12
ns = 1
nw = 6
nz = 3
g = 0.652
gprime = 0.352
yt = 0.995
muS = 1
lambdaHS = 0.1

def JF(x2, plus_minus):
    arr = []
    if plus_minus == 1:
        for x in x2:
            arr.append((integrate.quad(lambda y: (y * y *
             np.log(1 + np.exp(-((y * y + x) ** 0.5)))),
              0, np.inf))[0])

    elif plus_minus == -1:
        for x in x2:
            arr.append((integrate.quad(lambda y: (y * y *
             np.log(1 - np.exp(-((y * y + x) ** 0.5)))), 0,
              np.inf))[0])

    return np.array(arr)

def potential(T, phi):

    Vtree = ((lambdaH / 4) * (phi ** 2 - v ** 2)
     * (phi ** 2 - v ** 2))

    Mt2 = yt * yt * phi * phi / 2
    Mh2 = abs(3 * lambdaH * phi * phi - lambdaH * v * v)
    Ms2 = muS * muS + lambdaHS * phi * phi / 2
    Mw2 = g * g * phi * phi / 4
    Mz2 = (g * g + gprime * gprime) * phi * phi / 4

    Vcw1 = ((1 / (64 * np.pi * np.pi)) * (Mh2 ** 2)
     * (np.log(Mh2 / (v * v)) - (3 / 2)))

    Vcw2 = ((1 / (64 * np.pi * np.pi)) * (Ms2 ** 2)
     * (np.log(Ms2 / (v * v)) - (3 / 2)))
    
    Vcw3 = (6 * (1 / (64 * np.pi * np.pi)) * (Mw2 ** 2)
     * (np.log(Mw2 / (v * v)) - (5 / 6))
      + 3 * (1 / (64 * np.pi * np.pi)) * (Mz2 ** 2)
       * (np.log(Mz2 / (v * v)) - (5 / 6)))
    
    Vcw4 = (-12 * (1 / (64 * np.pi * np.pi)) * (Mt2 ** 2)
     * (np.log(Mt2 / (v * v)) - (3 / 2)))

    if T == 0:
        VT = np.zeros(Vtree.shape)
        VT0 = np.zeros(Vtree.shape)
    else:
        VT = (nt * (T ** 4) * JF(Mt2 / (T * T), 1)
         + nh * (T ** 4) * JF(Mh2 / (T * T), -1)
          + ns * (T ** 4) * JF(Ms2 / (T * T), -1)
           + nw * (T ** 4) * JF(Mw2 / (T * T), -1)
            + nz * (T ** 4) * JF(Mz2 / (T * T), -1))

        VT0 = (nt * (T ** 4)
         * JF([(yt * yt * 0.01 * 0.01 / 2) / (T * T)], 1)
         + nh * (T ** 4) * JF([(3 * lambdaH * 0.01 * 0.01
          -  lambdaH * v * v) / (T * T)], -1) + ns * (T ** 4)
           * JF([(muS * muS + lambdaHS * 0.01 * 0.01 / 2)
            / (T * T)], -1) + nw * (T ** 4)
             * JF([(g * g * 0.01 * 0.01 / 4)
             / (T * T)], -1) + nz * (T ** 4)
              * JF([((g * g + gprime * gprime)
              * 0.01 * 0.01 / 4) / (T * T)], -1)) 

    return ((Vtree + Vcw1 + Vcw2 + Vcw3 + Vcw4 + VT - VT0)
     / (v ** 4))

phi = np.linspace(0.01, 350, 3500)
for temp in [0, 50, 75]:
    potx = potential(temp, phi)
    plt.plot(phi / v, potx, label = ("V at T = " + str(temp) + " GeV"))
plt.xlabel('phi / v')
plt.ylabel('V(T, phi) / (v ** 4)')
plt.legend()
plt.savefig('singlet.png', dpi=800)
plt.show()
