clear all; clc; close all;
SDC_iterations = 4;
Newton_iterations = 3;
t0 = 0;
t_end = 10;
N = 100;
dt = (t_end-t0)/N;
a = -0.5;
f_e = @(t,u) a*u;
df_e = @(t,u) a;
f_i = @(t,u) 0*u;
df_i = @(t,u) 0;
u0 =  100 ;
U = zeros(N+1,1);
U(1) = u0;
t = t0;
M = 2; % #nodes = M+1
Lobatto_points = [-1,0,1];
Lobatto_weights = [1/3,4/3,1/3];
Lobatto_absolute_points = dt/2*Lobatto_points + dt/2; % in [0,dt]
dts = diff(Lobatto_absolute_points);
for n = 1:N
    U0 = ones(M+1,1)*U(n);
    ts = t + Lobatto_absolute_points;
    S_I = zeros(M,M+1);
    S_E = zeros(M,M+1);

    for i = 1:M
        for j = 1:M+1
            if j <= i
                S_E(i,j) = dts(j);
            end
            if 1< j && j <= i+1
                S_I(i,j) = dts(j-1);
            end
        end
    end
    Uk = U0;
    for k = 1:SDC_iterations
        F_E = f_e(ts,Uk);
        F_I = f_i(ts,Uk);
        I = zeros(M,1);
        y = polyfit(ts,F_E+F_I,M);
        Y = polyint(y);
        for i = 1:M
            I(i) = diff(polyval(Y,ts(i:i+1)));
        end
        S = cumsum(I);
        C = U0(2:end) -S_E*F_E -S_I*F_I + S;
        Uj = Uk;
        for j = 1:Newton_iterations
            grad = -S_E*df_e(ts,Uj) - S_I*df_i(ts,Uj);
            delta = grad\(C -S_E*f_e(ts,Uj)-S_I*f_i(ts,Uj));
            Uj = Uj+[0;delta(1:end-1)];
        end
        Uk(2:end) = C + S_E*f_e(ts,Uj) + S_I*f_i(ts,Uj) ;
    end
    U(n+1) = Uk(end);
    t = t+dt;
end
plot(0:dt:t_end, U)
hold on
actual = @(x) u0*exp(a*x);
fplot(actual, [t0,t_end])

