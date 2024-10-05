import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

MH = 125
MW = 81
MZ = 91
Mt = 173
lambda_ = 0.5

def potential(phi, T):
	A = (((MH ** 3) + 6 * (MW ** 3) + 3 * (MZ ** 3))
	 / (12 * np.pi * (phi ** 3)))

	D = ((MH ** 2 + 6 * MW ** 2 + 3 * MZ ** 2 + 6 
		* Mt ** 2) / (12 * (phi ** 2)))

	T0 = (MH / ((2 * D) ** 0.5))

	V = ((((D / 2) * (T * T - T0 * T0) * phi * phi) 
	- ((A / 3) * T * phi * phi * phi) 
	+ ((lambda_ / 4) * (phi ** 4))))

	return V

phi = np.linspace(0.01, 250, 2500)
for temp in [0, 75, 150]:
    potx = potential(phi, temp)
    plt.plot(phi, potx, label = ("V at T = " +
     str(temp) + " GeV"))

plt.xlabel('phi')
plt.ylabel('V(T, phi)')
plt.legend()
plt.savefig('standard.png', dpi=800)
plt.show()
