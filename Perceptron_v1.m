% Script pour simuler le fonctionnement d'un perceptron unique

clear all
close all
clc

% Paramètres de la méthode du gradient
nbItMax = 100;
rho(1) = 1;
abscisse = [1:1:nbItMax];

% Données d'entrainement
load('Programmes/DataSimulation/DataTrain_2Classes_Perceptron_2.mat');
N = length(data(1,:));
dim = length(data(:,1));
x = data;
c = c.';
x(dim+1,:) = ones(1,N);

% Données de test
load('Programmes/DataSimulation/DataTest_2Classes_Perceptron_2.mat');
xTest = dataTest;
cTest = cTest.';
xTest(dim+1,:) = ones(1,N);

% Initialisation des poids w
w(:,1) = [0.1;0.5;0.4];
%w(:,1) = 10*rand(3,1);

% Initialisation des autres grandeurs
y(:,1) = sigmoid((w(:,1).')*x);
f(1) = (1/(2*N))*sum((y(:,1)-c).^2);
gradf(:,1) = (1/N)*sum((x.').*(y(:,1)-c).*(y(:,1)).*(1-y(:,1)), 1);

% Boucle de calcul du gradient et mise à jour des paramètres
for ind = 2:nbItMax
    
    % Calcul des poids
    w(:,ind) = w(:,ind-1) - rho(ind-1)*gradf(:,ind-1);
    
    % Calcul de sorties
    y(:,ind) = sigmoid((w(:,ind).')*x);
    yTest(:,ind) = sigmoid((w(:,ind).')*xTest);
    
    % Calcul du gradient
    gradf(:,ind) = (1/N)*sum((x.').*(y(:,ind)-c).*(y(:,ind)).*(1-y(:,ind)), 1);
    
    % Calcul du crit�re
    f(ind) = (1/(2*N))*sum((y(:,ind)-c).^2);
    fTest(ind) = (1/(2*N))*sum((yTest(:,ind)-cTest).^2);
    
    % Ajustement du pas variable rho
    if (f(ind) < f(ind-1))
        rho(ind) = 2*rho(ind-1);
    else
        rho(ind) = rho(ind-1)/2;
        w(:,ind) = w(:,ind-1);
        f(ind) = f(ind-1);
    end
    
    % Comptage du nombre de données bien classifiées
    count(ind) = sum(y(:,ind)>(1/2) & (c == 1)) + sum(y(:,ind)<(1/2) & (c == 0));
    countTest(ind) = sum(yTest(:,ind)>(1/2) & (cTest == 1)) + sum(yTest(:,ind)<(1/2) & (cTest == 0));
    
end

% Affichage rho, norme du gradient et critère
figure(1);
title(['Evolution de f(w), de la norme du gradient de f(w) et de rho pour ',num2str(nbItMax),' itérations']);
subplot(311);
semilogy(abscisse, rho);
ylabel('rho');
subplot(312);
plot(abscisse, sqrt(sum(gradf.^2, 1)));
ylabel('Norme de gradf(w)');
subplot(313);
semilogy(abscisse, f);
ylabel('Critère f(w)');

% Nombre de points bien classifiés, critères données d'entraînement et test
figure(2);
title('Nombre de points bien classifiés et critère');
subplot(411);
plot(abscisse, count);
ylabel('Nb points bien classés Train');
subplot(412);
plot(abscisse, countTest);
ylabel('Nb points bien classés Test');
subplot(413);
semilogy(abscisse, f);
ylabel('Critère f(w)');
subplot(414);
semilogy(abscisse, fTest);
ylabel('Critère fTest(w)');

% Evolution des poids
figure(3);
title(['Evolution de w pour ',num2str(nbItMax),' iterations']);
plot(abscisse, w);
xlabel('Iterations');
ylabel('w');
legend('w1', 'w2', 'biais');

% Séparation des données
figure(4);
title(['Data Train - 2 Classes - Perceptron 2']);
print_2classes(x, c, y, nbItMax);


% Tracé de l'hyperplan

pts_abs = [-12 12];
pts_ord = -1/w(2,nbItMax)*(w(3,nbItMax)+w(1,nbItMax)*pts_abs);
plot(pts_abs, pts_ord, '-k');


%Données de test

% Séparation des données
figure(5);
title(['Data Test - 2 Classes - Perceptron 2']);
print_2classes(xTest, cTest, yTest, nbItMax);

% Tracé de l'hyperplan
pts_ord = -1/w(2,nbItMax)*(w(3,nbItMax)+w(1,nbItMax)*pts_abs);
plot(pts_abs, pts_ord, '-k');

