
function shrinkage_errors
    close all; clc; clear;
    shrink_symb={'X_lambda_star','X_s_star','X_optimal'};
    color = {'r','b','g'};
    dists = {'Gaussian','Uniform','Student-t(6)'};
    num_of_dists = length(dists);
    dims = [20 100];
    num_of_dims = length(dims);
    beta = 1;
    w = (8 * beta) ./ (beta + 1 + sqrt(beta.^2 + 14 * beta +1)); 
    lambdastar = sqrt(2 * (beta + 1) + w);
    % shrinkers: optimal shrinkage \ soft threshold \ hard threshold
    shrink{1} = @(x)(x .* (x>=lambdastar)); % 
    shrink{2} = @(x)( max(0, x-(1+sqrt(beta))));
    shrink{3} = @(x)( optshrink(x,beta)  );
  
    cut= [0 0 0];
    % Figutre 1: Plot with F-Norm
    fig1 = figure('Position',[25,25,1200,500]);
    fprintf('Figute 1 Plotting...\n');
    err_type = 0; % F-norm
    input_matrix = 0; % Randomal Sparse
    idx =1;
    for i=1:num_of_dims
        for j=1:num_of_dists
            fprintf('plot %d of %d\n',idx,num_of_dims*num_of_dists);
            subplot(num_of_dims,num_of_dists,idx);
            err_paper_origin(dims(i),dims(i),1,dists{j},shrink,shrink_symb,color,cut,input_matrix,err_type);
            idx = idx+1;
        end
    end
    % Figutre 2: Plot with 2-Norm
    fprintf('Figute 2 Plotting...\n');
    fig2 = figure('Position',[25,25,1200,500]);
    err_type = 1; % 2-norm
    input_matrix = 0; % Randomal Sparse
    idx =1;
    for i=1:num_of_dims
        for j=1:num_of_dists
            fprintf('plot %d of %d\n',idx,num_of_dims*num_of_dists);
            subplot(num_of_dims,num_of_dists,idx);
            err_paper_origin(dims(i),dims(i),1,dists{j},shrink,shrink_symb,color,cut,input_matrix,err_type);
            idx = idx+1;
        end
    end
    % Figutre 3: Plot with F-Norm and Toeplitz input matrix
    fprintf('Figute 3 Plotting...\n');
    fig3 = figure('Position',[25,25,1200,500]);
    err_type = 0; % F-norm
    input_matrix = 1; % Randomal Toeplitz
    idx =1;
    for i=1:num_of_dims
        for j=1:num_of_dists
            fprintf('plot %d of %d\n',idx,num_of_dims*num_of_dists);
            subplot(num_of_dims,num_of_dists,idx);
            err_paper_origin(dims(i),dims(i),0,dists{j},shrink,shrink_symb,color,cut,input_matrix,err_type);
            idx = idx+1;
        end
    end
    % Figutre 4: Plot with F-Norm different ranks
    fprintf('Figute 4 Plotting...\n');
    fig4 = figure('Position',[25,25,1200,500]);
    r_vals=[2 4];
    num_of_r_vals=length(r_vals);
    err_type = 0; % F-norm
    input_matrix = 0; % Randomal Sparse
    idx=1;  
    for i=1:num_of_dims
        for j=1:num_of_dists
            fprintf('plot %d of %d\n',idx,num_of_dims*num_of_dists);
            subplot(num_of_r_vals,num_of_dists,idx);
            err_paper_origin(50,50,r_vals(i),dists{j},shrink,shrink_symb,color,cut,input_matrix,err_type);
            idx = idx+1;
        end
    end

function result = optshrink(x,beta)
    % the optimal shrinker
    result = zeros(size(x));
    I = (x > 1+sqrt(beta));
    result(I) = sqrt((x(I).^2-beta-1).^2 -4*beta ) ./ x(I);

function err_paper_origin(M,N,r,dist,shrink,shrink_symb,color,cut,in_mat,error_type)
    dots = ['o','d','s'];
    mu_vals = 1:0.05:5; % must have mu>1
    nof_empirical = 50; % Monte Carlo
    mse = zeros(length(mu_vals), length(shrink));
    X = zeros(N,M);
    for i_mu = 1:length(mu_vals);
        mu = mu_vals(i_mu);
        if (in_mat==0)
            dX = zeros(1,M);
            dX(1:r) = mu;
            [UX tmp1 tmp2] = svd(randn(M));
            [tmp1 tmp2 VX] = svd(randn(N));
            X = UX * [diag(dX) zeros(M,N-M)] * VX';  
        elseif (in_mat==1)
            column = [mu,zeros(1,N-1)];
            row = [mu,zeros(1,M-1)];
            X=toeplitz(column,row);              
            r=rank(X);
        else
            printf ("Invalid in_mat\n")
        endif
        current_mse = zeros(nof_empirical,length(shrink)); 
        % Loop over monte carlo iterations
        for idx_empirical = 1:nof_empirical
            % generating noise
            switch dist
            case 'Student-t(6)'
                Z = trnd(6,M,N) / sqrt(6/4);
            case 'Gaussian'
                Z = randn(M, N);
            case 'Uniform'
                Z = (rand(M,N)-0.5)*sqrt(12);
            end
            Z = Z/sqrt(N);
            % data observation
            Y = X + Z; 
            [UY DY VY] = svd(Y);
            y = diag(DY);
            for shrink_idx =1:length(shrink)
                % shrink the data
                xhat = shrink{shrink_idx}(y);
                if(cut(shrink_idx))
                     xhat(r+1:end)=0;
                end
                % estimate the data from the shrinked noised observation
                DH = [diag(xhat) zeros(M, N - M)];
                H = UY * DH * VY'; 
                if (error_type==1)
                    current_mse(idx_empirical,shrink_idx)  = (norm(X - H,2));
                else
                    current_mse(idx_empirical,shrink_idx)  = (norm(X - H, 'fro')^2);
                endif 
            end
        end
        mse(i_mu,:) = mean(current_mse);
    end
    beta = M/N;
    I = mu_vals > beta^(0.25);
    t = zeros(size(mu_vals));
    w = zeros(size(mu_vals));
    t(I) = sqrt((mu_vals(I) + 1./mu_vals(I)).*(mu_vals(I) + beta./mu_vals(I)));
    t(~I) = 1+sqrt(beta);
    w(I) = (mu_vals(I).^4 - beta) ./ ((mu_vals(I).^2) .* t(I));
    for shrink_idx = 1:length(shrink)
    plot(mu_vals,mse(:,shrink_idx), [dots(shrink_idx) color{shrink_idx}], 'LineWidth',0.75,'MarkerSize',3);
    hold on;
    end
    for shrink_idx = 1:length(shrink)
    mse_f_assymptotic = zeros(size(mu_vals));
    mse_f_assymptotic = r * (shrink{shrink_idx}(t).^2 - 2 * shrink{shrink_idx}(t).* w + mu_vals.^2);
    plot(mu_vals,mse_f_assymptotic,['-' color{shrink_idx}],'LineWidth',0.75,'MarkerSize',3);
    end
    title(sprintf('(%d,%d,%d) %s',M,N,r,dist),'Interpreter','Latex','FontSize',14);
    axis tight;
    xlabel('x','FontSize',10,'Interpreter','Latex');
    ylabel('MSE','FontSize',10,'Interpreter','Latex');
    h=legend(shrink_symb,'Interpreter','Latex','Location','NorthWest');
    set(h,'FontSize',7);
