import Solvers
import numpy as np
import matplotlib.pyplot as plt

omega = 2

f = lambda tn,qn : -np.sin(qn)
g = lambda tn,pn: pn
h = lambda tn, yn: np.hstack([g(tn,yn[1]), f(tn,yn[0])])

H = lambda q,p: 1/2*(p**2 ) - np.cos(q)


dt = 1e-1
T = 100
ts = np.linspace(0,T,int(T/dt)+1)

symp = Solvers.SymplecticEuler(f,g,dt)
eul = Solvers.ForwardEuler(h,dt)

x0 = np.array([[0,1]])

end,symp_sol = symp.Step(x0[:,0],x0[:,1],0,T)
end, forw_sol = eul.Step(np.reshape(x0,2),0,T)

symp_q,symp_p = symp_sol
forw_q = forw_sol[:,0]
forw_p= forw_sol[:,1]


ax = plt.figure().add_subplot()
plt.title("Simulation")
ax.plot(ts,symp_q, label='Symplectic position')
ax.plot(ts,symp_p, label='Symplectic velocity')
ax.plot(ts,forw_q, label='Forward position')
ax.plot(ts,forw_p, label='Forward velocity')

ax.legend()
ax.set_xlabel('time')
ax.set_ylabel('phase')

ax = plt.figure().add_subplot()
plt.title("Hamiltonian")
symp_energy = H(symp_q,symp_p)
forw_energy = H(forw_q,forw_p)
ax.plot(ts,symp_energy, label='Symplectic Euler')
ax.plot(ts,forw_energy, label='Forward Euler')

ax.legend()
ax.set_xlabel('time')
ax.set_ylabel('H(q,p)')



plt.show()






