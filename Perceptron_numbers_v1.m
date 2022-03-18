% Script pour simuler le fonctionnement d'un perceptron unique

clear all
close all
clc

% Param�tres de la m�thode du gradient
nbItMax = 500;
rho(1) = 1;
abscisse = [1:1:nbItMax];

% Donn�es d'entrainement
% 1er chiffre
load('Programmes/Data/DigitTrain_1.mat');
dim = length(imgs(:,1,1)) * length(imgs(1,:,1));
N1 = length(imgs(1,1,:));
x1 = reshape(imgs,dim,N1);
c1 = labels;
p1 = 1;

% 2�me chiffre
load('Programmes/Data/DigitTrain_8.mat');
N2 = length(imgs(1,1,:));
x2 = reshape(imgs,dim,N2);
c2 = labels;

% Concatenation des exemples des deux classes
N = N1+N2;
x = cat(2,x1,x2);
x(dim+1,:) = ones(1,N);
c = cat(1,c1,c2);

% Donn�es de test
% 1er chiffre
load('Programmes/Data/DigitTest_1.mat');
NTest1 = length(imgs(1,1,:));
xTest1 = reshape(imgs,dim,NTest1);
cTest1 = labels;

% 2�me chiffre
load('Programmes/Data/DigitTest_8.mat');
NTest2 = length(imgs(1,1,:));
xTest2 = reshape(imgs,dim,NTest2);
cTest2 = labels;

% Concatenation des exemples des deux classes
NTest = NTest1+NTest2;
xTest = cat(2,xTest1,xTest2);
xTest(dim+1,:) = ones(1,NTest);
cTest = cat(1,cTest1,cTest2);

% Initialisation des poids w
%w(:,1) = ones(dim+1,1);
w(:,1) = 0.5*rand(dim+1,1);

% Initialisation des sorties y
y(:,1) = sigmoid((w(:,1).')*x);
yTest(:,1) = sigmoid((w(:,1).')*xTest);

% Initialisation des crit�res f et du nombre de donn�es bien classifi�es
% count
f(1) = (1/(2*N))*sum((y(:,1)-tp(p1,c)).^2);
count(1) = sum(y(:,1)>(1/2) & (tp(p1,c) == 1)) + sum(y(:,1)<(1/2) & (tp(p1,c) == 0));

fTest(1) = (1/(2*NTest))*sum((yTest(:,1)-tp(p1,cTest)).^2);
countTest(1) = sum(yTest(:,1)>(1/2) & (tp(p1,cTest) == 1)) + sum(yTest(:,1)<(1/2) & (tp(p1,cTest) == 0));

% Initialisation du gradient
gradf(:,1) = (1/N)*sum((x.').*(y(:,1)-tp(p1,c)).*(y(:,1)).*(1-y(:,1)), 1);

% Boucle de calcul du gradient et mise � jour des param�tres
for ind = 2:nbItMax
    
    % Calcul des poids
    w(:,ind) = w(:,ind-1) - rho(ind-1)*gradf(:,ind-1);
    
    % Calcul de sorties
    y(:,ind) = sigmoid((w(:,ind).')*x);
    yTest(:,ind) = sigmoid((w(:,ind).')*xTest);
    
    % Calcul du gradient
    gradf(:,ind) = (1/N)*sum((x.').*(y(:,ind-1)-tp(p1,c)).*(y(:,ind-1)).*(1-y(:,ind-1)), 1);
    
    % Calcul du crit�re
    f(ind) = (1/(2*N))*sum((y(:,ind)-tp(p1,c)).^2);
    fTest(ind) = (1/(2*N))*sum((yTest(:,ind)-tp(p1,cTest)).^2);
    
    % Ajustement du pas variable rho
    if (f(ind) < f(ind-1))
        rho(ind) = 2*rho(ind-1);
    else
        rho(ind) = rho(ind-1)/2;
        w(:,ind) = w(:,ind-1);
        f(ind) = f(ind-1);
    end
    
    % Comptage du nombre de donn�es bien classifi�es
    count(ind) = sum(y(:,ind)>(1/2) & (tp(p1,c) == 1)) + sum(y(:,ind)<(1/2) & (tp(p1,c) == 0));
    countTest(ind) = sum(yTest(:,ind)>(1/2) & (tp(p1,cTest) == 1)) + sum(yTest(:,ind)<(1/2) & (tp(p1,cTest) == 0));
    
end

% Affichage rho, norme du gradient et crit�re
figure(1);
title(['Evolution de f(w), de la norme du gradient de f(w) et de rho pour ',num2str(nbItMax),' it�rations']);
subplot(311);
semilogy(abscisse, rho);
ylabel('rho');
subplot(312);
plot(abscisse, sqrt(sum(gradf.^2, 1)));
ylabel('Norme de gradf(w)');
subplot(313);
semilogy(abscisse, f);
ylabel('Crit�re f(w)');

% Nombre de points bien classifi�s, crit�res donn�es d'entra�nement et test
figure(2);
title('Nombre de points bien classifi�s et crit�re');
subplot(411);
plot(abscisse, count);
ylabel('Nb points bien class�s Train');
subplot(412);
plot(abscisse, countTest);
ylabel('Nb points bien class�s Test');
subplot(413);
semilogy(abscisse, f);
ylabel('Crit�re f(w)');
subplot(414);
semilogy(abscisse, fTest);
ylabel('Crit�re fTest(w)');

% Evolution des poids
figure(3);
title(['Evolution de w pour ',num2str(nbItMax),' it�rations']);
plot(abscisse, w);
xlabel('It�rations');
ylabel('w');





