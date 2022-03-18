% x = data
% c = classe des data
% y = fonction sigmoid pour chaque donnée et chaque itération
% nbItMax = nb d'itérations max

function print_2classes(x, c, y, nbItMax)

    
load('DataTrain_2Classes_Perceptron.mat');
    
hold on;
cpt = 0;

for k = 1:length(c)
    if ( c(k) == 0 )
        if ( y(k,nbItMax) < 1/2 )
            plot( (x(1,k)), (x(2,k)), 'g+' );
            cpt = cpt + 1;
        else
            plot( (x(1,k)), (x(2,k)), 'r+' );
        end
    else
        if ( y(k,nbItMax) > 1/2 )
            plot( (x(1,k)), (x(2,k)), 'bd' );
            cpt = cpt + 1;
        else
            plot( (x(1,k)), (x(2,k)), 'rd' );
        end
    end
end
axis([-12 12 -12 12]);
cpt/length(c)*100