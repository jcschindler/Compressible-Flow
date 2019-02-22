################################################################################
#### Compressible Flow Relations  V_2.0  #######################################
##################################################### "Based on a True Story" ##
# by jeffy s ###################################################################
################################################################################
# Default for returning and printing outputs
returnValueDefault = False
printValueDefault = True

#############       These parameters are available on all functions
### Functions       , gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault)
#############

###########################
### Isentropic Relations
########################

#   ise.all(M,
#               - returns all stagnation ratios and the area ratio

#   ise.pressure(M,
#               - returns total pressure ratio p_total/p

#   ise.temperature(M,
#               - returns total temperature ratio T_total/T

#   ise.density(M,
#               - returns total density ratio rho_total/rho

#   ise.area(M,
#               - returns area ratio A/A_star (Quasi 1-D Flow)

#   ise.invArea(A/Astar ,domain,
#               - returns Mach number based on the given Area ratio A/Astar
#               - domain is either 'subsonic' or 'supersonic'

#   ise.invTemperature(T_total/T ,
#               - returns Mach number based on the given Temperature ratio T_total/T

#   ise.invPressure(p_total/p ,
#               - returns Mach number based on the given Pressure ratio p_total/p

#   ise.invDensity(rho_total/rho ,
#               - returns Mach number based on the given Density ratio rho_total/rho

#############################
### Normal Shock Relations
##########################
        # - Adiabatic
        # - Frictionless (still has losses in the shock)
        # - total temperature (T_o) is constant across a stationary shock wave.
        # - total pressure decreases across a shock wave.

# ns.postShockMach(M1,
#               - returns post-shock Mach number M2

# ns.density(M1,
#               - returns the density ratio rho2/rho1 across the shock

# ns.pressure(M1,
#               - returns the pressure ratio p2/p1 across the shock

# ns.temperature(M1,
#               - returns the temperature ratio T2/T1 across the shock

# ns.all(M1,
#				- returns all above normal shock relations

##############################################
### One Dimensional Flow With Heat Addition
###########################################
        # - The effect of heat addition is to directly change the total temperature of
        #		the flow
        # - Starred quantities in this section represent the condition of the fluid if
        #		it was accelerated to Mach 1 through heat addition

# heat.density(M1,
#               - returns the density ratio rho/rho* across the shock

# heat.pressure(M1,
#               - returns the pressure ratio p/p* across the shock

# heat.temperature(M1,
#               - returns the temperature ratio T/T* across the shock

# heat.allM1,
#				- returns ratio of conditions at Mach = M to Mach = 1 conditions

################################
### Frictional Flow Relations
#############################
        # - Starred quantities in this section represent the condition of the fluid if
        #		it was accelerated to Mach 1 through friction

# fric.density(M1,
#               - returns the density ratio rho/rho*

# fric.pressure(M1,
#               - returns the pressure ratio p/p*

# fric.temperature(M1,
#               - returns the temperature ratio T/T*

# fric.all(M1,
#				- returns ratio of conditions at Mach = M to to Mach = 1 conditions

#########################
### Prandtl Meyer Flow
######################

# pm.prandtl(M,
#               - returns the value of the Prandtl-Meyer function (v) at the given Mach number

# pm.invPrandtl(v,
#               - returns the Mach number at the given value of the Prandtl-Meyer function


################################
### Theta-Beta-Mach Relations
#############################
        #	Theta 	- Deflection Angle (degrees)
        #	Beta 	- Wave Angle (degrees)

# tbm.theta(M, beta,
#               - returns the value of theta for the given Mach number and angle beta

# tbm.beta(M, theta,
#               - returns the value of beta for the given Mach number and angle theta

# tbm.mach(theta, beta,
#               - returns the Mach number for the given angles theta and beta


import numpy as np
import math

# Isentropic Relations
class ise:
    def __init__(self):
        pass

    @staticmethod
    def pressure(M, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
        pressure0_1 = (1+((gamma-1)/2)*M**2)**(gamma/(gamma-1))
        if printval == True:
    	       print('p0/p1 =',format(pressure0_1,'.4f'))
        if returnval == True:
    	       return pressure0_1

    @staticmethod
    def temperature(M, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	Temperature0_1 = (1+((gamma-1)/2)*M**2)
    	if printval == True:
    		print('T0/T1 =',format(Temperature0_1,'.4f'))
    	if returnval == True:
    		return Temperature0_1

    @staticmethod
    def density(M, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	rho0_1 = (1+((gamma-1)/2)*M**2)**(1/(gamma-1))
    	if printval == True:
    		print('rho0/rho1 =',format(rho0_1,'.4f'))
    	if returnval == True:
    		return rho0_1

    @staticmethod
    def area(M, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	Aratio = math.sqrt((1/M**2)*((2/(gamma+1))*(1+(((gamma-1)/2)*(M**2))))**((gamma+1)/(gamma-1)))
    	if printval == True:
    		print('A/Astar =',format(Aratio,'.4f') )
    	if returnval == True:
    		return Aratio

    @staticmethod
    def invArea(Aratio, domain, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault): #range = 'subsonic' or 'supersonic' ie. inv_A_Astar(1.166,'subsonic')
    	if domain == 'subsonic':
    		M = [.0001,.001,.01,.1,1]
    	if domain == 'supersonic':
    		M = [1,3,5,7,51]
    	while(M[2]-M[1])>.00001:
    		nono = 0
    		vTest = []
    		vComp = []
    		for i in range(len(M)):
    			vTest.append(ise.area(M[i],gamma = gamma, returnval = True, printval = False))
    			vComp.append(vTest[i]-Aratio)
    			if domain == 'subsonic':
    				if vComp[i] < 0 and nono == 0:
    					val = i
    					nono = 1
    			if domain == 'supersonic':
    				if vComp[i] > 0 and nono == 0:
    					val = i
    					nono = 1
    		M = np.linspace(M[val-1],M[val],10)
    	if printval == True:
    	       print('M = ',format(M[val],'.4f'))
    	       print()
    	if returnval == True:
    	       return M

    @staticmethod
    def invTemperature(Tratio, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	M = [.00001,3,5,7,51]
    	while(M[2]-M[1])>.00001:
    		nono = 0
    		vTest = []
    		vComp = []
    		for i in range(len(M)):
    			vTest.append(ise.temperature(M[i],gamma = gamma, returnval = True, printval = False))
    			vComp.append(vTest[i]-Tratio)
    			if vComp[i] > 0 and nono == 0:
    				val = i
    				nono = 1
    		M = np.linspace(M[val-1],M[val],10)
    	if printval == True:
    	       print('M = ',format(M[val],'.4f'))
    	       print()
    	if returnval == True:
    	       return M

    @staticmethod
    def invPressure(Pratio, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault): #range = 'subsonic' or 'supersonic' ie. inv_A_Astar(1.166,'subsonic')
    	M = [.00001,3,5,7,51]
    	while(M[2]-M[1])>.00001:
    		nono = 0
    		vTest = []
    		vComp = []
    		for i in range(len(M)):
    			vTest.append(ise.pressure(M[i],gamma = gamma,  returnval = True, printval = False))
    			vComp.append(vTest[i]-Pratio)
    			if vComp[i] > 0 and nono == 0:
    				val = i
    				nono = 1
    		M = np.linspace(M[val-1],M[val],10)
    	if printval == True:
    	       print('M = ',format(M[val],'.4f'))
    	       print()
    	if returnval == True:
    	       return M

    @staticmethod
    def invDensity(Dratio, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	M = [.00001,3,5,7,51]
    	while(M[2]-M[1])>.00001:
    		nono = 0
    		vTest = []
    		vComp = []
    		for i in range(len(M)):
    			vTest.append(ise.density(M[i],gamma = gamma, returnval = True, printval = False))
    			vComp.append(vTest[i]-Dratio)
    			if vComp[i] > 0 and nono == 0:
    				val = i
    				nono = 1
    		M = np.linspace(M[val-1],M[val],10)
    	if printval == True:
    	       print('M = ',format(M[val],'.4f'))
    	       print()
    	if returnval == True:
    	       return M

    @staticmethod
    def all(M, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
        ise.pressure(M, gamma = gamma, returnval=returnval, printval=printval)
        ise.temperature(M, gamma = gamma, returnval=returnval, printval=printval)
        ise.density(M, gamma = gamma, returnval=returnval, printval=printval)
        ise.area(M, gamma = gamma, returnval=returnval, printval=printval)
        print()

# Normal Shock Relations
class ns:
    def __init__(self):
        pass

    @staticmethod
    def postShockMach(M1, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	num = 1 + (((gamma-1)/2) * M1**2)
    	den = (gamma*M1**2) - ((gamma-1)/2)
    	M2 = math.sqrt(num/den)
    	if printval == True:
    		print('M2 =', M2)
    	if returnval == True:
    		return M2

    @staticmethod
    def density(M1, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	num = (gamma+1) * M1**2
    	den = 2 +((gamma-1)*M1**2)
    	density2_1 = num/den
    	if printval == True:
    		print('rho2/rho1 =', density2_1)
    	if returnval == True:
    		return density2_1

    @staticmethod
    def pressure(M1,gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	pressure2_1 = 1 + ((2*gamma)/(gamma+1)) * (M1**2 - 1)
    	if printval == True:
    		print('p2/p1 =', pressure2_1)
    	if returnval == True:
    		return pressure2_1

    @staticmethod
    def temperature(M1,gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	pressure2_1 = ns.pressure(M1, gamma = gamma, returnval = True, printval = False)
    	density2_1 = ns.density(M1, gamma = gamma,returnval = True, printval = False)
    	temp2_1 = pressure2_1 * (1/density2_1)
    	if printval == True:
    		print('T2/T1 =', temp2_1)
    	if returnval == True:
    		return temp2_1

    @staticmethod
    def all(M1,gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	M2 = ns.postShockMach(M1,returnval = True, printval = False)
    	pressure2_1 = ns.pressure(M1, gamma = gamma,returnval = True, printval = False)
    	density2_1 = ns.density(M1, gamma = gamma,returnval = True, printval = False)
    	temp2_1 = ns.temperature(M1, gamma = gamma,returnval = True, printval = False)
    	if printval == True:
    		print('One-Dimensional Flow:')
    		print('	M1 =		', format(M1,'.4f'))
    		print('	p2/p1 =		', format(pressure2_1,'.4f'))
    		print('	rho2/rho1 =	', format(density2_1,'.4f'))
    		print('	T2/T1 =		', format(temp2_1,'.4f'))
    		print('	M2 =		', format(M2,'.4f'))
    		print('')
    	if returnval == True:
    		return M1, M2, density2_1, pressure2_1, temp2_1

# 1-D Flow with Heat Addition
class heat:
    def __init__(self):
        pass

    @staticmethod
    def density(M,gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	density2_1 = ((1+(gamma * M**2)) / (1+gamma))/M**2
    	if printval == True:
    		print('rho/rho* =', density2_1)
    	if returnval == True:
    		return density2_1

    @staticmethod
    def pressure(M,gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	pressure2_1 = (1+gamma)/(1+(gamma*M**2))
    	if printval == True:
    		print('p/p* =', pressure2_1)
    	if returnval == True:
    		return pressure2_1

    @staticmethod
    def temperature(M,gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	temp2_1 = M**2 * ((1+gamma)/(1+(gamma*M**2)))**2
    	if printval == True:
    		print('T/T* =', temp2_1)
    	if returnval == True:
    		return temp2_1

    @staticmethod
    def NormalTotalPressure_WithHeat(M,gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	pressureTot2_1 = ((1+gamma) / (1+(gamma*M**2))) * ((2+((gamma-1)*M**2))/(gamma+1))**(gamma/(gamma-1))
    	if printval == True:
    		print('p_o/p_o* =', temp2_1)
    	if returnval == True:
    		return pressureTot2_1

    @staticmethod
    def NormalTotalTemperature_WithHeat(M,gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	tempTot2_1 = (((gamma+1)*M**2)/(1+(gamma*M**2))**2) * (2 +((gamma-1)*M**2))
    	if printval == True:
    		print('T_o/T_o* =', temp2_1)
    	if returnval == True:
    		return tempTot2_1

    @staticmethod
    def all(M,gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	pressure2_1 = heat.pressure(M, gamma = gamma,returnval = True, printval = False)
    	density2_1 = heat.density(M, gamma = gamma,returnval = True, printval = False)
    	temp2_1 = heat.temperature(M, gamma = gamma,returnval = True, printval = False)
    	pressureTot2_1 = heat.NormalTotalPressure_WithHeat(M, gamma = gamma,returnval = True, printval = False)
    	tempTot2_1 = heat.NormalTotalTemperature_WithHeat(M, gamma = gamma,returnval = True, printval = False)
    	if printval == True:
    		print('One-Dimensional Flow with Heat Addition:')
    		print('	M =		', format(M,'.4f'))
    		print('	p/p* =		', format(pressure2_1,'.4f'))
    		print('	T/T* =		', format(temp2_1,'.4f'))
    		print('	rho/rho* =	', format(density2_1,'.4f'))
    		print('	p_o/p_o* = 	',format(pressureTot2_1,'.4f'))
    		print('	T_o/T_o* = 	',format(tempTot2_1,'.4f'))
    		print('')
    	if returnval == True:
    		return M, pressure2_1, temp2_1, density2_1, pressureTot2_1, tempTot2_1

# 1-D Flow with Friction
class fric:
    def __init__(self):
        pass

    @staticmethod
    def density(M, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	density2_1 = (((2+((gamma-1) * M**2)) / (1+gamma))**(1/2))/M
    	if printval == True:
    		print('rho/rho* =', density2_1)
    	if returnval == True:
    		return density2_1

    @staticmethod
    def pressure(M, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	pressure2_1 = (((1+gamma)/(2+((gamma-1)*M**2)))**(1/2))/M
    	if printval == True:
    		print('p/p* =', pressure2_1)
    	if returnval == True:
    		return pressure2_1

    @staticmethod
    def temperature(M, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	temp2_1 = ((gamma+1)/(2+((gamma-1)*M**2)))
    	if printval == True:
    		print('T/T* =', temp2_1)
    	if returnval == True:
    		return temp2_1

    @staticmethod
    def NormalTotalPressure_Friction(M, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	pressureTot2_1 = (((2+((gamma-1)*M**2))/(gamma+1))**((gamma+1)/(2*gamma-2)))/M
    	if printval == True:
    		print('p_o/p_o* =', temp2_1)
    	if returnval == True:
    		return pressureTot2_1

    @staticmethod
    def all(M, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	pressure2_1 = fric.pressure(M, gamma = gamma,returnval = True, printval = False)
    	density2_1 = fric.density(M, gamma = gamma,returnval = True, printval = False)
    	temp2_1 = fric.temperature(M, gamma = gamma,returnval = True, printval = False)
    	pressureTot2_1 = fric.NormalTotalPressure_Friction(M, gamma = gamma,returnval = True, printval = False)
    	if printval == True:
    		print('One-Dimensional Flow with Friction:')
    		print('	M =		', format(M,'.4f'))
    		print('	T/T* =		', format(temp2_1,'.4f'))
    		print('	p/p* =		', format(pressure2_1,'.4f'))
    		print('	rho/rho* =	', format(density2_1,'.4f'))
    		print('	p_o/p_o* = 	',format(pressureTot2_1,'.4f'))
    		print('')
    	if returnval == True:
    		return M, pressure2_1, temp2_1, density2_1, pressureTot2_1, tempTot2_1

# Prandtl-Meyer Flow
class pm:
    def __init__(self):
        pass

    @staticmethod
    def prandtl(M,gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	v = 180*(math.sqrt((gamma+1)/(gamma-1))*math.atan(math.sqrt(((gamma-1)/(gamma+1))*(M**2 - 1)))-math.atan(math.sqrt(M**2-1)))/math.pi
    	if printval == True:
    		print('v(M) =',format(v,'.4f') )
    	if returnval == True:
    		return v

    @staticmethod
    def invPrandtl(v, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	M = np.linspace(1,100,10) #
    	while(M[2]-M[1])>.00001:
    		nono = 0
    		vTest = []
    		vComp = []
    		for i in range(len(M)):
    			vTest.append(pm.prandtl(M[i], gamma = gamma,returnval = True, printval = False))
    			vComp.append(vTest[i]-v)
    			if vComp[i] > 0 and nono == 0:
    				val = i
    				nono = 1
    		M = np.linspace(M[val-1],M[val],10)
    	if printval == True:
    		print('M = ',format(M[val],'.4f'))
    	if returnval == True:
    		return M[val]

# Theta-Beta-Mach Relations
class tbm:
    def __init__(self):
        pass

    @staticmethod
    def theta(M, b, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	num = M**2 * (math.sin(math.pi*b/180)**2) - 1
    	den = M**2 *(gamma + math.cos(math.pi*2*b/180)) +2
    	theta = 180*math.atan(2*(1/(math.tan(math.pi*b/180)))*(num/den))/math.pi
    	if printval == True:
    		print('Theta =',format(theta,'.3f') )
    	if returnval == True:
    		return theta

    @staticmethod
    def beta(M, theta, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	B = np.linspace(1,90,10) # we know it is in this range (zero gives an error, if less than 1 use [.000001 .00001 .0001 .001 .01])
    	while(B[2]-B[1])>.00001:
    		nono = 0
    		thetaTest = []
    		thetaComp = []
    		for i in range(len(B)):
    			thetaTest.append(tbm.theta(M,B[i],returnval = True, printval = False))
    			thetaComp.append(thetaTest[i]-theta)
    			if thetaComp[i] > 0 and nono == 0:
    				val = i
    				nono = 1
    		B = np.linspace(B[val-1],B[val],10)
    	if printval == True:
    		print('beta = ',format(B[val],'.4f'))
    	if returnval == True:
    		return B[val]

    @staticmethod
    def mach(theta, beta, gamma = 1.4, returnval = returnValueDefault, printval = printValueDefault):
    	M = np.linspace(1,25,10) # we know it is in this range (zero gives an error, if less than 1 use [.000001 .00001 .0001 .001 .01])
    	while(M[2]-M[1])>.00001:
    		nono = 0
    		thetaTest = []
    		thetaComp = []
    		for i in range(len(M)):
    			thetaTest.append(tbm.theta(M[i],beta,returnval = True, printval = False))
    			thetaComp.append(thetaTest[i]-theta)
    			if thetaComp[i] > 0 and nono == 0:
    				val = i
    				nono = 1
    		M = np.linspace(M[val-1],M[val],10)
    	if printval == True:
    		print('M = ',format(M[val],'.4f'))
    	if returnval == True:
    		return M[val]
