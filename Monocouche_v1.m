% Script pour simuler le fonctionnement d'un réseau monocouche

clear all
close all
clc

% Paramètres de la méthode du gradient
nbItMax = 20;   % Nombre d'itérations


% Données d'entrainement
load('Data_Train_All.mat');
N = length(x(1,:));
dim = length(x(:,1));

% Données de test
load('Data_Test_All.mat');
NTest = length(xTest(1,:));

for p = 1:10
    
    rho(1,p) = 1;      % Rho initial
    
    % Initialisation des poids w
    %load('w_monocouche.mat');
    %w(:,1,:) = w_monocouche;
    w(:,1,p) = zeros(dim,1);
    %w(:,1,p) = 0.5*rand(dim,1);

    % Initialisation des sorties y
    y(:,1,p) = sigmoid((w(:,1,p).')*x);
    yTest(:,1,p) = sigmoid((w(:,1,p).')*xTest);

    % Initialisation des critères f et du nombre de données bien classifiées
    % count
    f(1,p) = (1/(2*N))*sum((y(:,1,p)-tp(p-1,c)).^2);
    %count(1,p) = sum(y(:,1,p)>(1/2) & (tp(p-1,c) == 1)) + sum(y(:,1,p)<(1/2) & (tp(p-1,c) == 0));

    fTest(1,p) = (1/(2*NTest))*sum((yTest(:,1,p)-tp(p-1,cTest)).^2);
    %countTest(1,p) = sum(yTest(:,1,p)>(1/2) & (tp(p-1,cTest) == 1)) + sum(yTest(:,1,p)<(1/2) & (tp(p-1,cTest) == 0));

    % Initialisation du gradient
    gradf(:,1,p) = (1/N)*sum((x.').*(y(:,1,p)-tp(p-1,c)).*(y(:,1,p)).*(1-y(:,1,p)), 1);
end

ind_mod = 2;
modulo = 10;
nb_display = 0;

% Boucle de calcul du gradient et mise à jour des paramètres
for ind = 2:nbItMax
    ind
    for p = 1:10
        % Calcul des poids
        w(:,ind_mod,p) = w(:,ind_mod-1,p) - rho(ind_mod-1,p)*gradf(:,ind_mod-1,p);

        % Calcul de sorties
        y(:,ind_mod,p) = sigmoid((w(:,ind_mod,p).')*x);
        yTest(:,ind_mod,p) = sigmoid((w(:,ind_mod,p).')*xTest);

        % Calcul du gradient
        gradf(:,ind_mod,p) = (1/N)*sum((x.').*(y(:,ind_mod,p)-tp(p-1,c)).*(y(:,ind_mod,p)).*(1-y(:,ind_mod,p)), 1);

        % Calcul du critère
        f(ind_mod,p) = (1/(2*N))*sum((y(:,ind_mod,p)-tp(p-1,c)).^2);
        fTest(ind_mod,p) = (1/(2*N))*sum((yTest(:,ind_mod,p)-tp(p-1,cTest)).^2);

        % Ajustement du pas variable rho
        if (f(ind_mod,p) < f(ind_mod-1,p))
            rho(ind_mod,p) = 2*rho(ind_mod-1,p);
        else
            rho(ind_mod,p) = rho(ind_mod-1,p)/2;
            w(:,ind_mod,p) = w(:,ind_mod-1,p);
            f(ind_mod,p) = f(ind_mod-1,p);
        end
    end
    
    % Comptage du nombre de données bien classifiées
    %count(ind_mod,p) = sum(y(:,ind_mod,p)>(1/2) & (tp(p-1,c) == 1)) + sum(y(:,ind_mod,p)<(1/2) & (tp(p-1,c) == 0));
    %countTest(ind_mod,p) = sum(yTest(:,ind_mod,p)>(1/2) & (tp(p-1,cTest) == 1)) + sum(yTest(:,ind_mod,p)<(1/2) & (tp(p-1,cTest) == 0));
    
    if (ind_mod == modulo)
        ind
        %if(ind >= nbItMax - modulo)
            abscisse = (nb_display*modulo)+1:(nb_display*modulo)+modulo;
            % Nombre de points bien classifiés, critères données d'entraînement et test
            figure(1)
            hold on
            subplot(511);
            %plot(abscisse, count/N, '-');
            ylabel('count');
            hold on
            subplot(512);
            %plot(abscisse, countTest/NTest, '-');
            ylabel('countTest');
            hold on
            subplot(513);
            hold on
            semilogy(abscisse, f(:,:), '-');
            ylabel('f(w)');
            subplot(514);
            hold on
            semilogy(abscisse, fTest(:,:), '-');
            ylabel('fTest(w)');
            subplot(515);
            semilogy(abscisse, rho(:,:), '-');
            ylabel('rho');
        %end
        % Les nouvelles valeurs deviennent les anciennes pour la prochaine
        % itération
        for p = 1:10
            w(:,1,p) = w(:,modulo,p);
            y(:,1,p) = y(:,modulo,p);
            yTest(:,1,p) = yTest(:,modulo,p);
            gradf(:,1,p) = gradf(:,modulo,p);
            f(1,p) = f(modulo,p);
            fTest(1,p) = fTest(modulo,p);
            rho(1,p) = rho(modulo,p);
        end
        %count(1) = count(modulo);
        %countTest(1) = countTest(modulo);
        
        ind_mod = 2;
        nb_display = nb_display + 1;
    else
        ind_mod = ind_mod + 1;
    end
    
    
end
[truc,alpha] = max(y(:,ind_mod-1,:), [], 3);
count = sum(tp(alpha-1,c));
count/N
[trucTest,alphaTest] = max(yTest(:,ind_mod-1,:), [], 3);
countTest = sum(tp(alphaTest-1,cTest));
countTest/NTest
