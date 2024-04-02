clear all; clc; close all;
nodes = 3;
iterations = 3;
t0 = 0;
t_end = 10;
N = 20;
dt = (t_end-t0)/N;
a = -0.5;
%du/dt = f(u(t))
f = @(t,y) a*y;
u0 = 100;
U = zeros(N+1,1);
U(1) = u0;
t = t0;
node_placement = [-1,0,1];
node_values = [1/3,4/3,1/3];
node_placement = dt/2*node_placement + dt/2;
for j = 1:N
    U_tilde = zeros(nodes,1);
    U_tilde(1) = U(j);
    ts = zeros(nodes,1);
    ts(1) = t;
    for i = 2:nodes
        dt_i = (node_placement(i)-node_placement(i-1));
        t = t+dt_i;
        ts(i) = t;
        U_tilde(i) =  U_tilde(i-1) + dt_i*f(t,U_tilde(i-1));
    end
    Y = zeros(iterations,nodes);
    E = zeros(nodes,1);
    for k = 1:iterations
        Y(k,:) = polyfit(ts,U_tilde,2);
        if j == 1
            Y_tilde = u0;
        else
            Y_tilde = polyval(Y(k,:),t-dt);
        end

        R = zeros(1,nodes);
        % (Lobatto nodes -> R(1) = 0)
        for i = 1:nodes
            R(i) = Y_tilde + integral(@(t) f(t,polyval(Y(k,:),t)),t-dt,ts(i)) - polyval(Y(k,:),ts(i));
        end
        for i = 2:nodes
            dt_i = (node_placement(i)-node_placement(i-1))*dt;
            E(i) = E(i-1) + dt_i*f(ts(i-1),E(i-1))+ (R(:,i) - R(:,i-1));
        end
        U_tilde = U_tilde + E;
    end
    U(j+1) = U_tilde(end);
end

plot(0:dt:t_end, U)
hold on
actual = @(x) 100*exp(a*x);
fplot(actual, [t0,t_end])
legend()
