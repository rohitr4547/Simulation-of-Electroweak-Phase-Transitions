import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def JF(x2, plus_minus):
    arr = []
    if plus_minus == 1:
        for x in x2:
            arr.append(-(integrate.quad(lambda y: (y * y
             * np.log(1 + np.exp(-((y * y + x) ** 0.5)))),
              0, np.inf))[0])
    elif plus_minus == -1:
        for x in x2:
            arr.append((integrate.quad(lambda y: (y * y
             * np.log(1 - np.exp(-((y * y + x) ** 0.5)))),
              0, np.inf))[0])
    return np.array(arr)

x = np.linspace(0, 400, 4000)
JF_ = JF(x, 1) 
JB_ = JF(x, -1) 
# plt.plot(x, JF_, label='JF(x)')
plt.plot(x, JB_, label='JB(x)')
plt.xlabel('x')
plt.ylabel('JB(x)')
plt.xscale('log')
plt.legend()
plt.savefig('oscillators_boson.png', dpi=800)
# plt.savefig('oscillators_fermion.png', dpi=800)
plt.show()
