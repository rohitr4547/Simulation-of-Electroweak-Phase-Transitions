import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import warnings
from math import sqrt
warnings.filterwarnings("ignore")

v = 246
lamb_h = 0.129
lamb_s = 0.1
nh = 1
n_plus_minus = 2
n_H = 1
n_A = 1
nw = 6
nz = 3
nt = -12
n_WL = 2
n_ZL = 1
n_gammaL = 1
g = 0.652
gprime = 0.352
yt = 0.995

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
            	np.log(1 - np.exp(-((y * y + x) ** 0.5)))),
            	 0, np.inf))[0])

    return np.array(arr)

def pot(phi, T, lamb_1, lamb_2, lamb_3, m1):

	PI_S = (((gprime ** 2) / 8) + ((g ** 2 + gprime ** 2) / 16)
	 + (lamb_s / 2) + (lamb_1 / 12) + ((lamb_1 + lamb_2 +
	  2 * lamb_3) / 24) +  ((lamb_1 + lamb_2 - 2 * lamb_3) / 24))

	PI_W = 2 * g * g * T * T
	PI_Y = 2 * gprime * gprime * T * T
	DelT2 = (((g * g * phi * phi / 4) + PI_W -
	 (gprime * gprime * phi * phi / 4)
	 - PI_Y) ** 2) + (g * g * gprime * gprime
	  * phi * phi * phi * phi / 4)

	Mh2 = 2 * lamb_h * phi * phi
	M_plus_minus2 = m1 ** 2 + lamb_1 * phi * phi / 2
	M_plus_minus2T = M_plus_minus2 + PI_S
	MH2 = abs(m1 ** 2 + (lamb_1 + lamb_2 + 2 * lamb_3)
	 * phi * phi / 2)

	MH2T = MH2 + PI_S
	MA2 = abs(m1 ** 2 + (lamb_1 + lamb_2 - 2 * lamb_3)
	 * phi * phi / 2)

	MA2T = MA2 + PI_S
	Mw2 = g * g * phi * phi / 4
	Mw2T = Mw2 + PI_W
	Mz2 = (g * g + gprime * gprime) * phi * phi / 4
	Mz2T = (Mz2 + PI_W + (DelT2 ** 0.5)) / 2
	Mt2 = yt * yt * phi * phi / 2
	M_gamma2 = Mz2
	M_gamma2T = (Mz2 + PI_W + PI_Y - (DelT2 ** 0.5)) / 2

	Vtree = np.array((lamb_h / 4) *
	 (((phi ** 2 - v ** 2) ** 2) - (v ** 4)))

	Vcw1 = ((1 / (64 * np.pi * np.pi)) * (Mh2 ** 2)
	 * (np.log(Mh2 / (v * v)) - (3 / 2)))

	Vcw2 = ((2 / (64 * np.pi * np.pi))
	 * (M_plus_minus2 ** 2) * (np.log(
	 	M_plus_minus2 / (v * v)) - (3 / 2)))

	Vcw3 = ((1 / (64 * np.pi * np.pi)) * (MH2 ** 2)
	 * (np.log(MH2 / (v * v)) - (3 / 2)))

	Vcw4 = ((1 / (64 * np.pi * np.pi)) * (MA2 ** 2)
	 * (np.log(MA2 / (v * v)) - (3 / 2)))

	Vcw5 = (6 * (1 / (64 * np.pi * np.pi))
	 * (Mw2 ** 2) * (np.log(Mw2 / (v * v))
	 - (5 / 6)) + 3 * (1 / (64 * np.pi * np.pi))
	  * (Mz2 ** 2) * (np.log(Mz2 / (v * v)) - (5 / 6)))

	Vcw6 = (-12 * (1 / (64 * np.pi * np.pi)) * (Mt2 ** 2)
	 * (np.log(Mt2 / (v * v)) - (3 / 2)))

	Vcw = (Vcw1 + Vcw2 + Vcw3 + Vcw4 + Vcw5 + Vcw6)

	Vdaisy = ((-T / (2 * np.pi * np.pi)) *
	 (n_plus_minus * ((M_plus_minus2T ** 1.5)
	 - (M_plus_minus2 ** 1.5)) + n_H *
	  ((MH2T ** 1.5) - (MH2 ** 1.5))
	  + n_A * ((MA2T ** 1.5) - (MA2 ** 1.5))
	   + n_WL * ((Mw2T ** 1.5) - (Mw2 ** 1.5))
	   + n_ZL * ((Mz2T ** 1.5) - (Mz2 ** 1.5))
	    + n_gammaL* ((M_gamma2T ** 1.5) - (M_gamma2 ** 1.5))))

	if T == 0:
		VT = np.zeros(Vtree.shape)
		VT0 = np.zeros(Vtree.shape)

	else:
		VT = (np.array((1 / (2 * np.pi * np.pi))
		 * (nt * (T ** 4) * JF(Mt2 / (T * T), 1)
		 + nh * (T ** 4) * JF(Mh2 / (T * T), -1)
		  + n_plus_minus * (T ** 4)
		   * JF(M_plus_minus2 / (T * T), -1)
		   + n_H * (T ** 4) * JF(MH2 / (T * T), -1)
		    + n_A * (T ** 4) * JF(MA2 / (T * T), -1)
		     + nw * (T ** 4) * JF(Mw2 / (T * T), -1)
		      + nz * (T ** 4) * JF(Mz2 / (T * T), -1))))

		VT0 = ((1 / (2 * np.pi * np.pi)) * (nt * (T ** 4)
		 * JF([(yt * yt * 0.01 * 0.01 / 2) / (T * T)], 1)
		  + nh * (T ** 4)
		   * JF([abs(3 * lamb_h * 0.01 * 0.01
		    -  lamb_h * v * v) / (T * T)], -1)
		    + n_plus_minus * (T ** 4)
		     * JF([(m1 ** 2 + lamb_1 * 0.01 * 0.01 / 2)
		     / (T * T)], -1) + n_H * (T ** 4)
		      * JF([(m1 ** 2 + (lamb_1 + lamb_2 + 2 * lamb_3)
		       * 0.01 * 0.01 / 2) / (T * T)], -1)
		       + n_A * (T ** 4) * JF([(m1 ** 2 +
		        (lamb_1 + lamb_2 - 2 * lamb_3)
		         * 0.01 * 0.01 / 2) / (T * T)], -1)
		         + nw * (T ** 4)
		          * JF([(g * g * 0.01 * 0.01 / 4)
		          / (T * T)], -1) + nz * (T ** 4)
		           * JF([((g * g + gprime * gprime)
		            * 0.01 * 0.01 / 4)
		           / (T * T)], -1)))

		VT0 = VT0 * np.ones(Vtree.shape)

	return ([(Vtree + Vcw + VT - VT0 + Vdaisy) / (v ** 4),
	 Vtree, Vcw, VT, VT0, Vdaisy])

phi = np.linspace(0.01, 350, 3500)
rng = np.random.default_rng()
l1s = []
l2s = []
l3s = []
ms = []
phics = []
Tcs = []
phibyTs = []
flag = 0

for test in range(1):
	print("Parameters used:")
	# l1 = rng.random() * 3 + 2
	# l2 = 0.5 * rng.random() - 0.5 * rng.random()
	# l3 = 1.5 * rng.random() - 1.5 * rng.random()
	# m = rng.integers(low=35, high=125)
	l1 = 5
	l2 = 0.1
	l3 = 1.5
	m = 80
	l1s.append(l1)
	l2s.append(l2)
	l3s.append(l3)
	ms.append(m)

	print("Lambda 1 =", round(l1, 6))
	print("Lambda 2 =", round(l2, 6))
	print("Lambda 3 =", round(l3, 6))
	print("M1 =", round(m, 6), "GeV")
	print("\n")
	print("Physical masses:")
	print("Field Dependent Mass of Charged Scalar Field S+- =",
	 round(sqrt(abs(m ** 2 + l1 * v * v / 2)), 6), "GeV")

	print("Field Dependent Mass of Neutral Scalar Field H =",
	 round(sqrt(abs(m ** 2
	  + (l1 + l2 + 2 * l3) * v * v / 2)), 6), "GeV")

	print("Field Dependent Mass of Neutral Scalar Field A =",
	 round(sqrt(abs(m ** 2
	  + (l1 + l2 - 2 * l3) * v * v / 2)), 6), "GeV")

	temp = 120
	add = 0.5
	print("\n")
	# print("Temperatures checked:")
	# while temp > 119.5 and temp < 140.5:
	# 	maxi = False 
	# 	potx = pot(phi, temp, l1, l2, l3, m)
	# 	for i in range(1, 2499):
	# 		if (potx[0][i + 1] < potx[0][i]
	# 		 and potx[0][i - 1] < potx[0][i]):
			
	# 			maxi = True

	# 		if (potx[0][i + 1] > potx[0][i]
	# 		 and potx[0][i - 1] > potx[0][i]
	# 		 and abs(potx[0][i] - potx[0][0])
	# 		  <= 1e-3 and maxi == True):

	# 			add = 0.12

	# 		if (potx[0][i + 1] > potx[0][i]
	# 		 and potx[0][i - 1] > potx[0][i]
	# 		 and abs(potx[0][i] - potx[0][0])
	# 		  <= 1e-4 and maxi == True):

	# 			add = 0.012

	# 		if (potx[0][i + 1] > potx[0][i]
	# 		 and potx[0][i - 1] > potx[0][i]
	# 		 and abs(potx[0][i] - potx[0][0])
	# 		  <= 1e-5 and maxi == True):

	# 			pot0_val = potx[0][i]
	# 			phic_val = phi[i]
	# 			phics.append(phic_val)
	# 			Tcs.append(temp)
	# 			phibyTs.append(phic_val / Tc_val[0])
	# 			flag = 1
	# 			plt.plot(phi / v, potx[0]
	# 				, label = "V at T = "
	# 			 + str(round(temp, 6)) + " GeV")

	# 			break
	# 	if flag == 1:
	# 		flag = 0
	# 		break
	# 	print(round(temp, 6), "GeV")
	# 	temp += add
	# 	add = 0.5
		
	# print("PHIC = ", round(phics[test], 6), "GeV")
	# print("TC = ", round(Tcs[test], 6), "GeV")
	# print("PHIC / TC = ", round(phibyTs[test], 6))

plt.xlabel('phi / v')
plt.ylabel('V(phi, T) / (v ** 4)')
plt.legend()
plt.savefig('doublet.png', dpi=800)
# plt.show()
