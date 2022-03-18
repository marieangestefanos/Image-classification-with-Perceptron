% Script pour simuler le fonctionnement d'un perceptron unique

clear all
close all
clc



% Paramètres de la méthode du gradient
nbItMax = 10;
rho(1) = 1;
abscisse = [1:1:nbItMax];

% Données d'entrainement
load('Programmes/DataSimulation/DataTrain_2Classes_Perceptron.mat');
N = length(data(1,:));
dim = length(data(:,1));
x = data;
c = c.';
x(dim+1,:) = ones(1,N);

% Données de test
load('Programmes/DataSimulation/DataTest_2Classes_Perceptron.mat');
xTest = dataTest;
cTest = cTest.';
xTest(dim+1,:) = ones(1,N);

% Initialisation des poids w
w(:,1) = [0.1;0.5;0.4];

% Initialisation des sorties y
y(:,1) = sigmoid((w(:,1).')*x);

yTest(:,1) = sigmoid((w(:,1).')*xTest);

% Initialisation des critères f et du nombre de données bien classifiées
% count
f(1) = (1/(2*N))*sum((y(:,1)-c).^2);
count(1) = sum(y(:,1)>1/2 & c == 1) + sum(y(:,1)<1/2 & c == 0);

fTest(1) = (1/(2*N))*sum((yTest(:,1)-cTest).^2);
countTest(1) = sum(yTest(:,1)>1/2 & cTest == 1) + sum(yTest(:,1)<1/2 & cTest == 0);

% Boucle de calcul du gradient et mise à jour des paramètres
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
title(['Evolution de f(w), de la norme du gradient de f(w) et de rho pour ',num2str(nbItMax),' itérations']);
subplot(311);
plot(abscisse, rho);
ylabel('rho');
subplot(312);
plot(abscisse, sqrt(sum(gradf.^2, 1)));
ylabel('Norme de gradf(w)');
subplot(313);
plot(abscisse, f);
ylabel('Critère f(w)');


figure(2);
title('Nombre de points bien classifiés et critère');
subplot(411);
plot(abscisse, count);
ylabel('Nb points bien classés Train');
subplot(412);
plot(abscisse, countTest);
ylabel('Nb points bien classés Test');
subplot(413);
plot(abscisse, f);
ylabel('Critère f(w)');
subplot(414);
plot(abscisse, fTest);
ylabel('Critère fTest(w)');

figure(3);
title(['Evolution de w pour ',num2str(nbItMax),' itérations']);
plot(abscisse, w);
xlabel('Itérations');
ylabel('w');
legend('w1', 'w2', 'biais');

figure(4);
print_2classes(data,c);
v1 = [0 -w(3,ind)*w(1,ind)];
v2 = [0 -w(3,ind)*w(2,ind)];
plot(v1, v2, '-k');


