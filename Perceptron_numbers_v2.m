% Script pour simuler le fonctionnement d'un perceptron unique

clear all
close all
clc

% Paramètres de la méthode du gradient
nbItMax = 500;
rho(1) = 1;

% Données d'entrainement
% 1er chiffre
load('Programmes/Data/DigitTrain_1.mat');
dim = length(imgs(:,1,1)) * length(imgs(1,:,1));
N1 = length(imgs(1,1,:));
x1 = reshape(imgs,dim,N1);
c1 = labels;
p1 = 1;

% 2ème chiffre
load('Programmes/Data/DigitTrain_8.mat');
N2 = length(imgs(1,1,:));
x2 = reshape(imgs,dim,N2);
c2 = labels;

% Concatenation des exemples des deux classes
N = N1+N2;
x = cat(2,x1,x2);
x(dim+1,:) = ones(1,N);
c = cat(1,c1,c2);

% Données de test
% 1er chiffre
load('Programmes/Data/DigitTest_1.mat');
NTest1 = length(imgs(1,1,:));
xTest1 = reshape(imgs,dim,NTest1);
cTest1 = labels;

% 2ème chiffre
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

% Initialisation des critères f et du nombre de données bien classifiées
% count
f(1) = (1/(2*N))*sum((y(:,1)-tp(p1,c)).^2);
count(1) = sum(y(:,1)>(1/2) & (tp(p1,c) == 1)) + sum(y(:,1)<(1/2) & (tp(p1,c) == 0));

fTest(1) = (1/(2*NTest))*sum((yTest(:,1)-tp(p1,cTest)).^2);
countTest(1) = sum(yTest(:,1)>(1/2) & (tp(p1,cTest) == 1)) + sum(yTest(:,1)<(1/2) & (tp(p1,cTest) == 0));

% Initialisation du gradient
gradf(:,1) = (1/N)*sum((x.').*(y(:,1)-tp(p1,c)).*(y(:,1)).*(1-y(:,1)), 1);

ind_mod = 2;
modulo = 50;
nb_display = 0;

% Boucle de calcul du gradient et mise à jour des paramètres
for ind = 2:nbItMax
    
    % Calcul des poids
    w(:,ind_mod) = w(:,ind_mod-1) - rho(ind_mod-1)*gradf(:,ind_mod-1);
    
    % Calcul de sorties
    y(:,ind_mod) = sigmoid((w(:,ind_mod).')*x);
    yTest(:,ind_mod) = sigmoid((w(:,ind_mod).')*xTest);
    
    % Calcul du gradient
    gradf(:,ind_mod) = (1/N)*sum((x.').*(y(:,ind_mod)-tp(p1,c)).*(y(:,ind_mod)).*(1-y(:,ind_mod)), 1);
    
    % Calcul du critère
    f(ind_mod) = (1/(2*N))*sum((y(:,ind_mod)-tp(p1,c)).^2);
    fTest(ind_mod) = (1/(2*N))*sum((yTest(:,ind_mod)-tp(p1,cTest)).^2);
    
    % Ajustement du pas variable rho
    if (f(ind_mod) < f(ind_mod-1))
        rho(ind_mod) = 2*rho(ind_mod-1);
    else
        rho(ind_mod) = rho(ind_mod-1)/2;
        w(:,ind_mod) = w(:,ind_mod-1);
        f(ind_mod) = f(ind_mod-1);
    end
    
    % Comptage du nombre de données bien classifiées
    count(ind_mod) = sum(y(:,ind_mod)>(1/2) & (tp(p1,c) == 1)) + sum(y(:,ind_mod)<(1/2) & (tp(p1,c) == 0));
    countTest(ind_mod) = sum(yTest(:,ind_mod)>(1/2) & (tp(p1,cTest) == 1)) + sum(yTest(:,ind_mod)<(1/2) & (tp(p1,cTest) == 0));
    
    if (ind_mod == modulo)
        if(ind >= nbItMax - modulo)
            abscisse = (nb_display*modulo)+1:(nb_display*modulo)+modulo;
            % Nombre de points bien classifiés, critères données d'entraînement et test
            figure(1)
            hold on
            subplot(511);
            plot(abscisse, count/N, '-');
            ylabel('count');
            hold on
            subplot(512);
            plot(abscisse, countTest/NTest, '-');
            ylabel('countTest');
            hold on
            subplot(513);
            hold on
            semilogy(abscisse, f, '-');
            ylabel('f(w)');
            subplot(514);
            hold on
            semilogy(abscisse, fTest, '-');
            ylabel('fTest(w)');
            subplot(515);
            semilogy(abscisse, rho, '-');
            ylabel('rho');
        end
        % Les nouvelles valeurs deviennent les anciennes pour la prochaine
        % itération
        w(:,1) = w(:,modulo);
        y(:,1) = y(:,modulo);
        yTest(:,1) = yTest(:,modulo);
        gradf(:,1) = gradf(:,modulo);
        f(1) = f(modulo);
        fTest(1) = fTest(modulo);
        rho(1) = rho(modulo);
        count(1) = count(modulo);
        countTest(1) = countTest(modulo);
        
        ind_mod = 2;
        nb_display = nb_display + 1;
    else
        ind_mod = ind_mod + 1;
    end

end

% Evolution des poids
%     figure(3);
%     hold on;
%     title(['Evolution de w pour ',num2str(nbItMax),' itérations']);
%     plot(ind, w);
%     xlabel('Itérations');
%     ylabel('w');





