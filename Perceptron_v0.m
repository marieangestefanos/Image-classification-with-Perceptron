% Script pour simuler le fonctionnement d'un perceptron unique

clear all
close all
clc



% Param�tres de la m�thode du gradient
nbItMax = 10;
rho(1) = 1;
abscisse = [1:1:nbItMax];

% Donn�es d'entrainement
load('Programmes/DataSimulation/DataTrain_2Classes_Perceptron.mat');
N = length(data(1,:));
dim = length(data(:,1));
x = data;
c = c.';
x(dim+1,:) = ones(1,N);

% Donn�es de test
load('Programmes/DataSimulation/DataTest_2Classes_Perceptron.mat');
xTest = dataTest;
cTest = cTest.';
xTest(dim+1,:) = ones(1,N);

% Initialisation des poids w
w(:,1) = [0.1;0.5;0.4];

% Initialisation des sorties y
y(:,1) = sigmoid((w(:,1).')*x);

yTest(:,1) = sigmoid((w(:,1).')*xTest);

% Initialisation des crit�res f et du nombre de donn�es bien classifi�es
% count
f(1) = (1/(2*N))*sum((y(:,1)-c).^2);
count(1) = sum(y(:,1)>1/2 & c == 1) + sum(y(:,1)<1/2 & c == 0);

fTest(1) = (1/(2*N))*sum((yTest(:,1)-cTest).^2);
countTest(1) = sum(yTest(:,1)>1/2 & cTest == 1) + sum(yTest(:,1)<1/2 & cTest == 0);

% Boucle de calcul du gradient et mise � jour des param�tres
for ind = 2:nbItMax
    
    % Calcul du gradient
    gradf(:,ind) = (1/N)*sum((x.').*(y(:,ind-1)-c).*(y(:,ind-1)).*(1-y(:,ind-1)), 1);
    w(:,ind) = w(:,ind-1) - rho(ind-1)*gradf(:,ind);
    y(:,ind) = sigmoid((w(:,ind).')*x);
    f(ind) = (1/(2*N))*sum((y(:,ind)-c).^2);
    
    yTest(:,ind) = sigmoid((w(:,ind).')*xTest);
    fTest(ind) = (1/(2*N))*sum((yTest(:,ind)-cTest).^2);
    
    if (f(ind) <= f(ind-1))
        rho(ind) = 2*rho(ind-1);
    else
        rho(ind) = rho(ind-1)/2;
        w(:,ind) = w(:,ind-1);
        f(ind) = f(ind-1);
    end
    
    count(ind) = sum(y(:,ind)>1/2 & c == 1) + sum(y(:,ind)<1/2 & c == 0);
    countTest(ind) = sum(yTest(:,ind)>1/2 & cTest == 1) + sum(yTest(:,ind)<1/2 & cTest == 0);
    
end

figure(1);
title(['Evolution de f(w), de la norme du gradient de f(w) et de rho pour ',num2str(nbItMax),' it�rations']);
subplot(311);
plot(abscisse, rho);
ylabel('rho');
subplot(312);
plot(abscisse, sqrt(sum(gradf.^2, 1)));
ylabel('Norme de gradf(w)');
subplot(313);
plot(abscisse, f);
ylabel('Crit�re f(w)');


figure(2);
title('Nombre de points bien classifi�s et crit�re');
subplot(411);
plot(abscisse, count);
ylabel('Nb points bien class�s Train');
subplot(412);
plot(abscisse, countTest);
ylabel('Nb points bien class�s Test');
subplot(413);
plot(abscisse, f);
ylabel('Crit�re f(w)');
subplot(414);
plot(abscisse, fTest);
ylabel('Crit�re fTest(w)');

figure(3);
title(['Evolution de w pour ',num2str(nbItMax),' it�rations']);
plot(abscisse, w);
xlabel('It�rations');
ylabel('w');
legend('w1', 'w2', 'biais');

figure(4);
print_2classes(data,c);
v1 = [0 -w(3,ind)*w(1,ind)];
v2 = [0 -w(3,ind)*w(2,ind)];
plot(v1, v2, '-k');


