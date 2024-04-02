Bx = 10.4;
By = 20.623;
Bz = 12.41095;
vx = 2345.4;
vy = 1235.623;
vz = 38.41095;
beta = 2852.234;
B = [Bx;By;Bz];
IxB = [0,   Bz, -By;
      -Bz,  0,   Bx;
       By, -Bx,  0];
v = [vx; vy; vz];
disp(["(IxB)*v == vxB", all(IxB*v==cross(v,B))])

I = eye(3);

A = I- beta*IxB;

alpha = 1/(1+norm(beta*B)^2)*(I + beta*IxB + beta^2*(B*B'));
alpha2 = 1/(1+norm(beta*B)^2)*(I - beta*IxB + beta^2*(B*B'));

alpha*A
disp(["Using + gives correct inverse?", norm(eye(3)-alpha*A) < 1e-10])
alpha2*A
disp(["Using - gives correct inverse?", norm(eye(3)-alpha2*A) < 1e-10])
alpha2'*A
disp(["Using transpose of - gives correct inverse?", norm(eye(3)-alpha2'*A) < 1e-10])