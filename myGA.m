% Credit: codes developed in this project have been inspired by the following references:
% [1] Pramanik, Niranjan, “Genetic Algorithm—explained step by step with example,” towardsdatascience.com, para. all, Sep. 9, 2019. [Online]. Available: https://towardsdatascience.com/genetic-algorithm-explained-step-by-step-65358abe2bf. [Accessed Nov. 19, 2019].


function [fxbest,xbest] = myGA(funchandle, xrange)
% INPUTS:
% xrange    : 1-by-d vector with ranges of each element of an input value
% funchandle: the handle of the function being optimized
%             (e.g., funchandle = @dejong)
% OUTPUTS
% fxbest: best fitness function value
% xbest : associated best value of the input vector x (column vector)
% Note that the internal code should automatically (or deterministically)
% define the encoding for the genotype and termination criteria


% STEP 1: generate the initial population of chromosomes x
%         where each element in x(i,:) is in [-xrange(k)/2 : xrange(k)/2]
%         i.e. random values distributed over the interval
%         for which the value of fitness function funchandle is calculated
% parameters: dimension of each chromosome x, population size of chomosomes
[~, d]   = size(xrange);
pop_size = 100;
% initial population
xold        = zeros(pop_size,d);
% initial population in binary form is xold_b
% initial population in Gray code form is xold_g
% generate initial population
for k = 1:d
    a = -xrange(k)/2;
    b =  xrange(k)/2;
    % create an array of random floating-point numbers that are drawn from 
    % a uniform distribution in the open interval ] -xrange/2 : xrange/2 [
    xold(:,k)       = (b-a).*rand(pop_size,1,'double') + a;
end
% convert decimal numbers to signed binary
q = quantizer('double');
xold_b = cellstr( num2bin(q,xold) );
xold_b = reshape( xold_b,[pop_size,d] );
% convert signed binary to gray code
xold_g = bin2gray( xold_b );
% initial population is generated
cur_gen = 1;
% repeat Select/Reproduce/Mutate for a number of max_gen generations
max_gen = 50;
% save to plot fmin vs generation number; preallocate space:
X = ones(max_gen,1);
Y = ones(max_gen,1);
while cur_gen <= max_gen
    
    disp('current generation:'); disp(cur_gen);
    
    % STEP 2: perform reproduction/selection using 'roulette wheel selection'
    %         this is a minimization problem, i.e.
    %         a chromosome that produces a low fitness value (low funchandle)
    %         has high fitness probability
    % fx    = fitness value of ith chromosome
    % f     = fitness value of ith chromosome with scaling
    % fnorm = fitness probability of ith chromosome
    % e.g., funchandle  = @dejong
    % at the minimum point, the fitness function has a min value
    % argmin f = argmax -f
    % The solution for min(f(x)) is the same as the solution for max(-f(x))
    fx = @(x) - funchandle(x);    
    % *********************************************************************
    % Scaling: using fx(x) - fmin results in an all positive function
    % fmin is the smallest value of the function fx(x) for the current 
    % population. So each time we find a new value fx(x) < old fmin using
    % a new population, we set fmin to this new value
    % *********************************************************************
    fmin  = min(fx(xold));
    % save to plot fmin vs generation number
    X(cur_gen) = cur_gen;
    Y(cur_gen) = fmin;
    % scaling
    f     = @(x) fx(x) - fmin;
    fnorm = f(xold) / sum(f(xold));
    % check if fitness probabilities sum to 1
    fnorm_cum = cumsum(fnorm);
    % for pop_size times, generate random values of
    % fitness probabilities from 0 to 1
    % create an array of random floating-point numbers that are drawn from 
    % a uniform distribution in the open interval ] 0 : 1 [
    pnorm_cum = rand(pop_size,1,'double');
    % indices of selected chromosomes
    selection = zeros(pop_size,1);
    % for each i, find the first index for which fnorm_cum > pnorm_cum(i)
    for i = 1:pop_size
        selection(i) = find(fnorm_cum > pnorm_cum(i), 1);
    end
    % new population from selected chromosomes
    xnew = xold( selection,: );
    % new population in binary form   
    xnew_b = xold_b( selection,: );
    % new population in Gray code form
    xnew_g = xold_g( selection,: );


    % STEP 3: perform crossover by exchanging portions of parents bits
    % crossover probability: This parameter value lies between 0 and 1
    % A value of 75% indicates that, out of a total of pop_size chromosomes
    % 75% are allowed to crossover to produce an equal number of offsprings
    p_crossover = 0.75;
    tot_parents = p_crossover * pop_size;
    % round down to nearest EVEN integer: 2 parents needed in each crossover
    isNotEven   = mod(tot_parents,2)>=1;
    tot_parents = floor(tot_parents);
    tot_parents(isNotEven) = tot_parents(isNotEven)+1;
    % randomly pair off chromosomes in population for crossover
    % totaling tot_parents
    % randperm is used to ensure each chromosome is only paired once
    % indices of pairs to undergo crossover stored in pairsCrOv
    % remaining chromosomes (tot_parents+1):pop_size will be kept the same
    lst       = randperm(pop_size);
    pairsCrOv = lst(1:tot_parents);
    xcrov_g   = xnew_g(pairsCrOv,:);
    % randomly select 2 crossoverpoints for each pair
    % pointslst is a (tot_parents/2)-by-2 column vector containing integer  
    % values drawn from a discrete uniform distribution whose range is 
    % *******************************************************************
    % each element of x(i) is double (elemlength=64 bit), with d elements in x
    elemlength = 64;
    chromlength = elemlength*d;
    pointslst = randi(chromlength,(tot_parents/2),2);
    % these crossover points are for each chromosome in a pair of two parents
    % hence duplicate each point in a row on the next row
    pointsCrOv = zeros(tot_parents,2);
    idx = 1:tot_parents;
    % isodd = rem(idx, 2) ~= 0;
    pointsCrOv(idx(rem(idx, 2) ~= 0),:) = pointslst;
    % iseven = rem(idx, 2) == 0;
    pointsCrOv(idx(rem(idx, 2) == 0),:) = pointslst;
    % Finally, exchange portions of strings between crossover points 
    % pointsCrOv(i,1) and pointsCrOv(i,2)
    % the Gray code forms will be crossed-over (crossover on binary forms
    % results in large changes in value)
    for i=1:2:tot_parents
        % join the d strings (of elemlength bits each) in each chromosome
        C1 = xcrov_g(i,:);
        C2 = xcrov_g(i+1,:);
        P1 = strjoin(C1,'');
        P2 = strjoin(C2,'');
        % crossover
        if pointsCrOv(i,1) <= pointsCrOv(i,2)
            strCrOv1 = P1(pointsCrOv(i,1):pointsCrOv(i,2));
            strCrOv2 = P2(pointsCrOv(i,1):pointsCrOv(i,2));
            P2(pointsCrOv(i,1):pointsCrOv(i,2)) = strCrOv1;
            P1(pointsCrOv(i,1):pointsCrOv(i,2)) = strCrOv2;
        else
            strCrOv1 = P1([1:pointsCrOv(i,2) , pointsCrOv(i,1):chromlength]);
            strCrOv2 = P2([1:pointsCrOv(i,2) , pointsCrOv(i,1):chromlength]);
            P2([1:pointsCrOv(i,2) , pointsCrOv(i,1):chromlength]) = strCrOv1;
            P1([1:pointsCrOv(i,2) , pointsCrOv(i,1):chromlength]) = strCrOv2;
        end
        % *******************************************************************
        % this deletes COPIES of parents values, replaces them with children values
        % split into d strings (of elemlength bits each) in each chromosome to
        % update vector xcrov_g with new offsprings / children
        xcrov_g(i,:)   = ( cellstr(reshape(P1,elemlength,[])') )';
        xcrov_g(i+1,:) = ( cellstr(reshape(P2,elemlength,[])') )';
    end
    % convert Gray code to signed binary
    xcrov_b = gray2bin(xcrov_g);
    % convert signed binary to decimal numbers
    q       = quantizer('double');
    xcrov   = cell2mat ( bin2num( q , xcrov_b) );
    % *******************************************************************
    % this deletes ORIGINAL values of parents, replaces them with children values
    % update the total population with the new offsprings/children
    % indices of pairs to undergo crossover stored in pairsCrOv
    % remaining chromosomes (tot_parents+1):pop_size will be kept the same
    % remember: lst = randperm(pop_size); pairsCrOv = lst(1:tot_parents);
    xnew_g(pairsCrOv,:) = xcrov_g;
    xnew_b(pairsCrOv,:) = xcrov_b;
    xnew(pairsCrOv,:)   = xcrov;
    
    
    % STEP 4: Mutation
    % The mutation parameter decides how many genes to be mutated
    % If mutation parameter is p_mutation = 0.001 (usually kept very low)
    % Then p_mutation times the total genes are allowed to mutate
    % In the present optimization problem, total number of genes is:
    % gene_tot  = (pop_size * d * elemlength) = (100 x 7 x 64)
    % Therefore, the number of genes allowed for mutation is:
    % gene_mut  = p_mutation * (pop_size * d * elemlength)
    % Regarding mutation positions, gene_mut random values of rows and columns 
    % are chosen. We need to randomly select gene_num positions where 
    % the gene's value (0 or 1) is to be flipped (to 1 or 0)
    p_mutation = 0.001;
    gene_tot   = (pop_size * d * elemlength);
    gene_mut   = ceil ( p_mutation * gene_tot );
    % locate gene: generate random indices in population, in dimension, in element
    % gene_pop_idx is a gene_mut-by-1 column vector containing integer values drawn from a 
    % discrete uniform distribution whose range is 1,2,...,pop_size
    % Similarly for gene_dim_idx and gene_elem_idx
    gene_pop_idx  = randi(pop_size,gene_mut,1);
    gene_dim_idx  = randi(d,gene_mut,1);
    gene_elem_idx = randi(elemlength,gene_mut,1);
    % flip value in gene
    for k=1:gene_mut
        % copy dimension of chromosome containing the gene
        charcopy = xnew_g{gene_pop_idx(k),gene_dim_idx(k)};
        % flip the value of that particular gene
        charcopy(gene_elem_idx(k)) = num2str( 1 - str2double(charcopy(gene_elem_idx(k))) );
        % update the mutated value in the population
        xnew_g(gene_pop_idx(k),gene_dim_idx(k)) = cellstr( charcopy );
        % convert Gray code to signed binary
        xnew_b(gene_pop_idx(k),gene_dim_idx(k)) = gray2bin( xnew_g(gene_pop_idx(k),gene_dim_idx(k)) );
        % convert signed binary to decimal numbers
        q                                       = quantizer('double');
        xnew(gene_pop_idx(k),gene_dim_idx(k))   = cell2mat( bin2num( q , xnew_b(gene_pop_idx(k),gene_dim_idx(k)) ) );
    end
    
    
    % ADDITIONAL STEP: Select/Reproduce/Mutate may have resulted in values
    % that are outside the range defined by xrange -> bring these values to 
    % closest range boundary
    % Repeat copies of min array: -xrange/2
    minRange  = repmat(-xrange/2,pop_size,1);
    % locate indices where the values is less than min values -xrange/2
    ind       = xnew<minRange;
    % assign to those values the corresponding min values -xrange/2
    xnew(ind) = minRange(ind);    
    % Repeat copies of max array: xrange/2
    maxRange = repmat(xrange/2,pop_size,1);
    % locate indices where the values is more than max values xrange/2
    ind       = xnew>maxRange;
    % assign to those values the corresponding max values xrange/2
    xnew(ind) = maxRange(ind);
    % convert decimal numbers to signed binary
    q = quantizer('double');
    xnew_b = cellstr( num2bin(q,xnew) );
    xnew_b = reshape( xnew_b,[pop_size,d] );
    % convert signed binary to gray code
    xnew_g = bin2gray( xnew_b );
    
    
    % ADDITIONAL STEP: replace NaN (Not a Number) elements in xnew
    % locates indices of NaN values in xnew
    TF = isnan(xnew);
    % index into xnew with TF to access the elements of xnew that are NaN.
    % replace the NaN values with 0.
    xnew(TF) = 0;
    
    
    % ADDITIONAL STEP: Elitism
    % replace the least fit chromosome in xnew
    % by the most fit chromosome from xold
    % most fit chromosome in xold maximizes fx (remember fx = -funchundle):
    % same chromosome could be repeated; pick it once only
    fxold_best  = max(fx(xold));
    idx_best    = find( fx(xold)==fxold_best,1 );
    xold_best   = xold( idx_best,: );
    % least fit chromosome in xnew minimizes fx (remember fx = -funchundle):
    % same chromosome could be repeated; pick it once only
    fxnew_worst = min(fx(xnew));
    idx_worst   = find( fx(xnew)==fxnew_worst,1 );
    % xnew_worst  = xnew( idx_worst,: );
    % assign the elite chromosome value from xold to least fit chrom in xnew
    xnew( idx_worst,: ) = xold_best;
    
    
    % FINAL STEP: suppress old generation with new generation
    xold    = xnew;
    xold_b  = xnew_b;
    xold_g  = xnew_g;
    cur_gen = cur_gen+1;
end


% most fit chromosome in xold maximizes fx (remember fx = -funchundle):
fxbest = max(fx(xold));
% find xbest:
% same chromosome could be repeated; pick it once only
idx_best    = find( fx(xold)==fxbest,1 );
xbest       = xold( idx_best,: );
% find fxbest:
% correct value for fxbest (remember fx = -funchundle)
fxbest = -fxbest;


% plot membership functions:
plt = Plot(X, Y);
% change settings
plt.XLabel = 'Generation Number';   % xlabel
plt.YLabel = 'Best f_x(x)'; % ylabel
plt.Legend = ["Best f_x(x) in each generation"];

