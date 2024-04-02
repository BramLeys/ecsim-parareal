%% Integration matrix construction by calculation
clear all; clc;close all;
n = 3;
fx = @(x) exp(x);
syms y
x = sym("x"	,[n,1]);
f = sym("f"	,[n,1]);
c = sym("c"	,[n,1]);
w = sym("w", [n,1]);
P = sym("P", [n,1]);
P_int = sym("P_int", [n,1]);
coe = sym("coe", [n,1]);
exact_c = sym("c_ex", [n,1]);
for i = 1:n
    P(i) = legendreP(i-1,y);
    P_int(i) = int(P(i),y);
    c(i) = (2*(i-1)+1)/2*w'*(f.*subs(P(i),y,x));
    exact_c_int = int(exp(y)*P(i),y);
    exact_c(i) = (2*(i-1)+1)/2*(subs(exact_c_int,y,1) - subs(exact_c_int,y,-1));
end

t = zeros(n,1);
t(1) = -1;
t(end) = 1;
r = sort(vpa(root(diff(P(end),y),y)));
for i = 2:n-1
    t(i) = r(i-1);
end

a = 0;
b = 1;
h = b-a;
dt = h/2*t + (a+b)/2;
m = 1;
for i = 2:n-1
    coe(i) = 2/(n*(n-1)*subs(P(end),y,t(i)).^2);
end
coe(1) = 2/(n*(n-1));
coe(n) = coe(1);
% w = [w1; w2; w3];
% y = [x1; x2; x3];
% f = [f1;f2;f3];

A = sym("A", [n-1,1]);
S = sym("S", [n-1,n]);
S(:,:) = 0;
for j = 1:n-1
    A(j) = (dt(j+1)-dt(j))/(t(j+1)-t(j))*sum(c.*subs(P_int,y,x(j+1)) - c.*subs(P_int,y,x(j)));
    for i = 1:n
        S(j,:) = S(j,:) + ((2*(i-1)+1)/2*(w.*subs(P(i),y,x))')*(subs(P_int(i),y,x(j+1))-subs(P_int(i),y,x(j)));
    end
    S(j,:) = S(j,:)*(dt(j+1)-dt(j))/(t(j+1)-t(j));
end
[K,b] = equationsToMatrix(A,f);
d = latex(K(1,1));
e = latex(S(1,1));

plot(t,exp(dt))
figure 
plot(t,subs(subs(c'*P,[w,f,x],[coe,fx(dt),t]),y,dt))
hold on
plot(t,subs(subs(exact_c'*P,[w,f,x],[coe,fx(dt),t]),y,dt))
plot(t,exp(dt))

% Found integral
found = eval(vpa(subs(A(m),[w,f,x], [coe,fx(dt),t])))
% Actual integral
actual = integral(@(x) fx(x), dt(m),dt(m+1))
% Matrix integral
matrix = eval(vpa(subs(K*f, [w,f,x], [coe,fx(dt),t])))
% polynomial integral
B = zeros(n);
for i = 1:n
    for j = 1:n
        B(i,j) = dt(i)^(j-1);
    end
end
% A = [1,t(1), t(1)^2;1,t(2), t(2)^2;1,t(3), t(3)^2;];
coeff = flip(B\fx(dt))';
q = polyint(coeff);
plot(t,polyval(coeff,dt))
legend("Interpolation", "Interpolation with exact integral", "Actual function","Vandermonde interpolation")
I = diff(polyval(q,[t(1),t(2)]))

%% integration matrix construction by derivation
close all; clc;clear all;
n = 3;
syms y
for i = 1:n
    P(i) = legendreP(i-1,y);
    P_int(i) = int(P(i),y);
end
t = zeros(n,1);
t(1) = -1;
t(end) = 1;
r = sort(vpa(root(diff(P(end),y),y)));
for i = 2:n-1
    t(i) = r(i-1);
end
w = zeros(n,1);
for i = 2:n-1
    w(i) = 2/(n*(n-1)*subs(P(end),y,t(i)).^2);
end
w(1) = 2/(n*(n-1));
w(n) = w(1);
% S
S_standard = zeros(n-1,n);
for j = 1:n-1
    for i = 1:n
        S_standard(j,:) = S_standard(j,:) + ((2*(i-1)+1)/2*(w.*subs(P(i),y,t))')*(subs(P_int(i),y,t(j+1))-subs(P_int(i),y,t(j)));
    end
end
a = 0;
b = a + 0.5;
lambda = -2;
h = b-a;
dt = h/2*t + (a+b)/2;
f = @(t,x) lambda*x;
% actual = zeros(n-1,1);
S = zeros(n-1,n);
for i = 1:n-1
    S(i,:) = (dt(i+1)-dt(i))/(t(i+1)-t(i))*S_standard(i,:);
    actual(i) = integral(@(t) f(t,10), dt(i),dt(i+1));
end
res = S*f(dt)
actual


%% explicit SDC
close all; clc;clear all;
n = 10;
syms y
for i = 1:n
    P(i) = legendreP(i-1,y);
    P_int(i) = int(P(i),y);
end
t = zeros(n,1);
t(1) = -1;
t(end) = 1;
r = sort(vpa(root(diff(P(end),y),y)));
for i = 2:n-1
    t(i) = r(i-1);
end
w = zeros(n,1);
for i = 2:n-1
    w(i) = 2/(n*(n-1)*subs(P(end),y,t(i)).^2);
end
w(1) = 2/(n*(n-1));
w(n) = w(1);
% S
S_standard = zeros(n-1,n);
for j = 1:n-1
    for i = 1:n
        S_standard(j,:) = S_standard(j,:) + ((2*(i-1)+1)/2*(w.*subs(P(i),y,t))')*(subs(P_int(i),y,t(j+1))-subs(P_int(i),y,t(j)));
    end
end
K = 10;
error = zeros(10,1);
iteration_error = zeros(K,1);
for timestep = 1:10
    a = 0;
    b = a + 2.^(-timestep);
    lambda = -2;
    h = b-a;
    dt = h/2*t + (a+b)/2;
    f = @(t,x) lambda*x;
    analytical_f = @(t) 10*exp(lambda*t);
    % actual = zeros(n-1,1);
    S = zeros(n-1,n);
    for i = 1:n-1
        S(i,:) = (dt(i+1)-dt(i))/(t(i+1)-t(i))*S_standard(i,:);
        % actual(i) = integral(f, dt(i),dt(i+1));
    end
    % res = S*F
    % actual
    
    U0 = 10;
    U = zeros(K,n);
    U(1,:) = U0;
    U(:,1) = U0;
    for k = 1:K-1
        F = f(dt,U(k,:)');
        integr = S*F;
        for m = 1:n-1
            U(k+1,m+1) = U(k+1,m) + (dt(m+1)-dt(m))*(f(dt(m),U(k+1,m)) - f(dt(m), U(k,m))) + integr(m);
        end
        iteration_error(k) = norm(U(k,:)-analytical_f(dt)');
    end
    error(timestep) = norm(U(end,:) - analytical_f(dt)');
    figure
    semilogy(1:K,iteration_error)
    title("Convergence with increasing SDC iterations, dt = "+2^-timestep)
end
semilogy(1:10,error)
title("Convergence with decreasing dt")
figure
semilogy(dt,U(end,:))
hold on
semilogy(dt,analytical_f(dt))
legend("calculated", "analytical")
%% Implicit SDC
close all; clc;clear all;
n = 5;
syms y
for i = 1:n
    P(i) = legendreP(i-1,y);
    P_int(i) = int(P(i),y);
end
t = zeros(n,1);
t(1) = -1;
t(end) = 1;
r = sort(vpa(root(diff(P(end),y),y)));
for i = 2:n-1
    t(i) = r(i-1);
end
w = zeros(n,1);
for i = 2:n-1
    w(i) = 2/(n*(n-1)*subs(P(end),y,t(i)).^2);
end
w(1) = 2/(n*(n-1));
w(n) = w(1);
% S
S_standard = zeros(n-1,n);
for j = 1:n-1
    for i = 1:n
        S_standard(j,:) = S_standard(j,:) + ((2*(i-1)+1)/2*(w.*subs(P(i),y,t))')*(subs(P_int(i),y,t(j+1))-subs(P_int(i),y,t(j)));
    end
end

K = 10;
error = zeros(10,1);
iteration_error = zeros(K,1);
for timestep = 1:10
    a = 0;
    b = a + 2.^(-timestep);
    lambda = -2;
    h = b-a;
    dt = h/2*t + (a+b)/2;

    % f_I = @(t,x) lambda*x;
    % df_I = @(t,x) lambda^2*x;
    % analytical_f_I = @(t) 5*exp(lambda*t);
    f_I = @(t,x) zeros(size(x));
    df_I = @(t,x) zeros(size(x));
    analytical_f_I = @(t) zeros(size(t));
    f_E = @(t,x) lambda/20*x;
    df_E = @(t,x) ones(size(x))*(lambda/20);
    analytical_f_E = @(t) 5*exp(lambda/20*t);
    analytical_f = @(t)  analytical_f_E(t) + analytical_f_I(t);
    U0 = analytical_f_I(0) + analytical_f_E(0);

    Sigma = zeros(n,n);
    SI = zeros(n);
    SE = zeros(n);
    for i = 2:n
        Sigma(i,:) = Sigma(i-1,:)+(dt(i)-dt(i-1))/(t(i)-t(i-1))*S_standard(i-1,:);
        SI(i,:) = SI(i-1,:);
        SI(i,i) = dt(i)-dt(i-1);
        SE(i,:) = SE(i-1,:);
        SE(i,i-1) = dt(i)-dt(i-1);
    end

    U = zeros(K,n);
    U(1,:) = U0;
    U(:,1) = U0;
    for k = 1:K-1
        F_I = f_I(dt,U(k,:)');
        F_E = f_E(dt,U(k,:)');
        g = @(x) SI*f_I(0,x) + SE*f_E(0,x) + U(1,:)' + Sigma*(F_I + F_E) - SI*F_I - SE*F_E;
        dg = @(x) SI*df_I(0,x) + SE*df_E(0,x);
        U(k+1,:) = newton(U(k,:)',g,dg,10);
        iteration_error(k) = norm(U(k,:)-analytical_f(dt)');
    end
    error(timestep) = norm(U(end,:) - analytical_f(dt)');
end
semilogy(1:10,error)
figure
semilogy(dt,U(end,:))
hold on
semilogy(dt,analytical_f(dt))
legend("calculated", "analytical")
figure
semilogy(1:K,iteration_error)

function [U1] = newton(U0,f,df, iterations)
    U1 = U0;
    for i = 1: iterations
        delta = f(U1)./df(U1);
        U1(2:end) = U1(2:end) + delta(2:end);
    end
end