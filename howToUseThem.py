import compressibleFlow as cf

# look in 'compressibleFlow.py' for details about function outputs:

# Isentropic Relations
# --use-->   cf.ise.
    # pressure(M)
    # density(M)
    # temperature(M)
    # area(M)
    # all(M)
    # invPressure(p_0/p)
    # invDensity(rho_0/rho)
    # invTemperature(T_0/T)
    # invArea(A/A*)
# Normal Shock Relations
# --use-->   cf.ns.
    # postShockMach(M1)
    # pressure(M1)
    # density(M1)
    # temperature (M1)
    # all(M1)
# 1-D Flow with Heat Addition
# --use-->   cf.heat.
    # pressure(M1)
    # density(M1)
    # temperature(M1)
    # all(M1)
# 1-D Flow with Friction
# --use-->   cf.heat.
    # pressure(M1)
    # density(M1)
    # temperature(M1)
    # all(M1)
# Prandtl-Meyer Flow
# --use-->   cf.pm.
    # prandtl(M)
    # invPrandtl(v)
# Theta-Beta-Mach Relations
# --use-->   cf.tbm.
    # theta(M,beta)
    # beta(M,theta)
    # mach (theta,beta)


cf.ise.all(2)
cf.ise.invArea(1.6875,'supersonic')
cf.ise.invArea(1.6875,'subsonic', gamma = 1.33)
cf.ise.invDensity(4.3469)
cf.ise.invPressure(7.8244)
cf.ise.invTemperature(1.8000)


# Store outputs for later use :)
po_p2 = cf.ise.pressure(2,gamma = 700,returnval = True, printval = False)
print(po_p2)

# you can adjust gamma, returnval, and printval on any function
# default:
#   gamma = 1.4
#   returnval = False
#   printval  = True

#   note:
#       you can adjust the default for printval and returnval easily in compressibleFlow.py





# let me know if you have any suggestions for how
# to make this a more enjoyable experience :)

# Jeff S.
# 781 956 6656 (cell)
# jcschind@uvm.edu
# jeff.c.schindler@gmail.com
# jcschindler17@gmail.com
# /u/sparklebubblez29_5
