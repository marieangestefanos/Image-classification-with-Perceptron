% Script pour concaténer les données avec toutes les images

clear all
close all
clc


% Données d'entrainement

% 0ème chiffre
load('Programmes/Data/DigitTrain_0.mat');
dim = length(imgs(:,1,1)) * length(imgs(1,:,1));
N0 = length(imgs(1,1,:));
x0 = reshape(imgs,dim,N0);
c0 = labels;

% 1er chiffre
load('Programmes/Data/DigitTrain_1.mat');
N1 = length(imgs(1,1,:));
x1 = reshape(imgs,dim,N1);
c1 = labels;

% 2ème chiffre
load('Programmes/Data/DigitTrain_2.mat');
N2 = length(imgs(1,1,:));
x2 = reshape(imgs,dim,N2);
c2 = labels;

% 3ème chiffre
load('Programmes/Data/DigitTrain_3.mat');
N3 = length(imgs(1,1,:));
x3 = reshape(imgs,dim,N3);
c3 = labels;

% 4ème chiffre
load('Programmes/Data/DigitTrain_4.mat');
N4 = length(imgs(1,1,:));
x4 = reshape(imgs,dim,N4);
c4 = labels;

% 5ème chiffre
load('Programmes/Data/DigitTrain_5.mat');
N5 = length(imgs(1,1,:));
x5 = reshape(imgs,dim,N5);
c5 = labels;

% 6ème chiffre
load('Programmes/Data/DigitTrain_6.mat');
N6 = length(imgs(1,1,:));
x6 = reshape(imgs,dim,N6);
c6 = labels;

% 7ème chiffre
load('Programmes/Data/DigitTrain_7.mat');
N7 = length(imgs(1,1,:));
x7 = reshape(imgs,dim,N7);
c7 = labels;

% 8ème chiffre
load('Programmes/Data/DigitTrain_8.mat');
N8 = length(imgs(1,1,:));
x8 = reshape(imgs,dim,N8);
c8 = labels;

% 9ème chiffre
load('Programmes/Data/DigitTrain_9.mat');
N9 = length(imgs(1,1,:));
x9 = reshape(imgs,dim,N9);
c9 = labels;

% Concatenation des exemples des deux classes
N = N0+N1+N2+N3+N4+N5+N6+N7+N8+N9;
x = cat(2,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9);
x(dim+1,:) = ones(1,N);
c = cat(1,c0,c1,c2,c3,c4,c5,c6,c7,c8,c9);

save('Data_Train_All.mat', 'x', 'c');

% Données de test

% 0ème chiffre
load('Programmes/Data/DigitTest_0.mat');
NTest0 = length(imgs(1,1,:));
xTest0 = reshape(imgs,dim,NTest0);
cTest0 = labels;

% 1er chiffre
load('Programmes/Data/DigitTest_1.mat');
NTest1 = length(imgs(1,1,:));
xTest1 = reshape(imgs,dim,NTest1);
cTest1 = labels;

% 2ème chiffre
load('Programmes/Data/DigitTest_2.mat');
NTest2 = length(imgs(1,1,:));
xTest2 = reshape(imgs,dim,NTest2);
cTest2 = labels;

% 3ème chiffre
load('Programmes/Data/DigitTest_3.mat');
NTest3 = length(imgs(1,1,:));
xTest3 = reshape(imgs,dim,NTest3);
cTest3 = labels;

% 4ème chiffre
load('Programmes/Data/DigitTest_4.mat');
NTest4 = length(imgs(1,1,:));
xTest4 = reshape(imgs,dim,NTest4);
cTest4 = labels;

% 5ème chiffre
load('Programmes/Data/DigitTest_5.mat');
NTest5 = length(imgs(1,1,:));
xTest5 = reshape(imgs,dim,NTest5);
cTest5 = labels;

% 6ème chiffre
load('Programmes/Data/DigitTest_6.mat');
NTest6 = length(imgs(1,1,:));
xTest6 = reshape(imgs,dim,NTest6);
cTest6 = labels;

% 7ème chiffre
load('Programmes/Data/DigitTest_7.mat');
NTest7 = length(imgs(1,1,:));
xTest7 = reshape(imgs,dim,NTest7);
cTest7 = labels;

% 8ème chiffre
load('Programmes/Data/DigitTest_8.mat');
NTest8 = length(imgs(1,1,:));
xTest8 = reshape(imgs,dim,NTest8);
cTest8 = labels;

% 9ème chiffre
load('Programmes/Data/DigitTest_9.mat');
NTest9 = length(imgs(1,1,:));
xTest9 = reshape(imgs,dim,NTest9);
cTest9 = labels;

% Concatenation des exemples des deux classes
NTest = NTest0+NTest1+NTest2+NTest3+NTest4+NTest5+NTest6+NTest7+NTest8+NTest9;
xTest = cat(2,xTest0,xTest1,xTest2,xTest3,xTest4,xTest5,xTest6,xTest7,xTest8,xTest9);
xTest(dim+1,:) = ones(1,NTest);
cTest = cat(1,cTest0,cTest1,cTest2,cTest3,cTest4,cTest5,cTest6,cTest7,cTest8,cTest9);

save('Data_Test_All.mat', 'xTest', 'cTest');