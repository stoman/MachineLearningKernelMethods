%This file implements the Nyström method for reducing the size of the Gram
%matrix. It is used for the classification problem using kernel methods.
%Author: Stefan Toman (toman@tum.de)

%input file
file = '../data/twoD_large_veryhard.mat';
%size of the new smaller matrix (should be a constant multiple of the rank
%of the Gram matrix)
samplesize = 25;

%load functions
addpath('../functions');

%kernel
h = 1;
gamma = 1;
K = gaussiankernel(gamma/h^2,2);

%load data file
load(file);

%the Nyström decomposition can be computed like this (which is done in the
%function funpredictnystrom))
%[A, B] = createnystrom(X, K, samplesize, true);

predict = funpredictnystrom(X, Y, K, samplesize);
predictions = predict(Xt);
%fprintf('%d of %d correct\n', predictionquality(predict, Xt, Yt), size(Xt,1));

%visualize outputs
if size(X,2) == 1
    %one-dimensional data
    %sample prediction function
    x = linspace(min(X),max(X),50)';
    z = predict(x);
    %create some plots
    scatter(subplot(3,2,1),X,zeros(size(X)),25,Y,'filled'); title('Training Data');
    scatter(subplot(3,2,2),Xt,zeros(size(Xt)),25,Yt,'filled'); title('Test Data');
    stem(subplot(3,2,3), x, z); title('Prediction Function'); colorbar();
    scatter(subplot(3,2,4),Xt,zeros(size(Xt)),25,predictions,'filled'); title('Predictions');  colorbar();
    ax = subplot(3,2,5);
    text(0, 0.9, sprintf('works for %d of %d inputs\n', sum(Yt - sign(predictions) == 0), size(Xt, 1)));
    text(0, 0.6, sprintf('%d training inputs\n', size(X,1)));
    text(0, 0.3, sprintf('h = %d, gamma = %d, sample size = %d\n', h, gamma, samplesize));
    text(0, 0.0, sprintf('data file %s\n', strrep(file, '_', '\_')));
    set(ax, 'visible', 'off');
    scatter(subplot(3,2,6),Xt,zeros(size(Xt)),25,sign(predictions),'filled'); title('Rounded Predictions');
elseif size(X,2) == 2
    %two-dimensional data
    %sample prediction function
    f = @(x,y) reshape(predict([reshape(x,[],1) reshape(y,[],1)]),size(x)); 
    [x, y] = meshgrid(linspace(min(X(:,1)),max(X(:,1)),100),linspace(min(X(:,2)),max(X(:,2)),100));
    z = f(x, y);
    %create some plots
    scatter(subplot(3,2,1),X(:,1),X(:,2),25,Y,'filled'); title('Training Data');
    scatter(subplot(3,2,2),Xt(:,1),Xt(:,2),25,Yt,'filled'); title('Test Data');
    contourf(subplot(3,2,3), x, y, z); title('Prediction Function'); colorbar();
    scatter(subplot(3,2,4),Xt(:,1),Xt(:,2),25,predictions,'filled'); title('Predictions'); colorbar();
    ax = subplot(3,2,5);
    text(0, 0.9, sprintf('works for %d of %d inputs\n', sum(Yt - sign(predictions) == 0), size(Xt, 1)));
    text(0, 0.6, sprintf('%d training inputs\n', size(X,1)));
    text(0, 0.3, sprintf('h = %d, gamma = %d, sample size = %d\n', h, gamma, samplesize));
    text(0, 0.0, sprintf('data file %s\n', strrep(file, '_', '\_')));
    set(ax, 'visible', 'off');
    scatter(subplot(3,2,6),Xt(:,1),Xt(:,2),25,sign(predictions),'filled'); title('Rounded Predictions');
else
    %high-dimensional data
    fprintf('dimension %d is too large to plot\n', size(Xt, 2));
    fprintf('works for %d of %d inputs\n', sum(Yt - sign(predictions) == 0), size(Xt, 1));
end
