file = 'twoD_small_veryhard.mat';
lambda = 1;
gamma = 1;
%K = @(x,z) x*z';
K = @(x,z) exp(-gamma.*(bsxfun(@plus, sum(x.^2,2), sum(z.^2,2)') - 2*(x*z')));

load(file);
a = (pdist2(X, X, K) + lambda*eye(size(X, 1)))\Y;
predict = @(Xt) (pdist2(Xt, X, K) * a);
predictions = predict(Xt);

if size(X,2) == 1
    x = linspace(min(X),max(X),50)';
    z = predict(x);

    scatter(subplot(3,2,1),X,zeros(size(X)),25,Y,'filled'); title('Training Data');
    scatter(subplot(3,2,2),Xt,zeros(size(Xt)),25,Yt,'filled'); title('Test Data');
    stem(subplot(3,2,3), x, z); title('Prediction Function');
    scatter(subplot(3,2,4),Xt,zeros(size(Xt)),25,predictions,'filled'); title('Predictions');
    ax = subplot(3,2,5);
    text(0, 0.9, sprintf('works for %d of %d inputs\n', sum(Yt - sign(predictions) == 0), size(Xt, 1)));
    text(0, 0.6, sprintf('%d training inputs\n', size(X,1)));
    text(0, 0.3, sprintf('lambda = %d, gamma = %d\n', lambda, gamma));
    text(0, 0.0, sprintf('data file %s\n', strrep(file, '_', '\_')));
    set(ax, 'visible', 'off');
    scatter(subplot(3,2,6),Xt,zeros(size(Xt)),25,sign(predictions),'filled'); title('Rounded Predictions');
elseif size(X,2) == 2
    f = @(x,y) reshape(predict([reshape(x,[],1) reshape(y,[],1)]),size(x)); 
    [x, y] = meshgrid(linspace(min(X(:,1)),max(X(:,1)),100),linspace(min(X(:,2)),max(X(:,2)),100));
    z = f(x, y);

    scatter(subplot(3,2,1),X(:,1),X(:,2),25,Y,'filled'); title('Training Data');
    scatter(subplot(3,2,2),Xt(:,1),Xt(:,2),25,Yt,'filled'); title('Test Data');
    contourf(subplot(3,2,3), x, y, z); title('Prediction Function');
    scatter(subplot(3,2,4),Xt(:,1),Xt(:,2),25,predictions,'filled'); title('Predictions');
    ax = subplot(3,2,5);
    text(0, 0.9, sprintf('works for %d of %d inputs\n', sum(Yt - sign(predictions) == 0), size(Xt, 1)));
    text(0, 0.6, sprintf('%d training inputs\n', size(X,1)));
    text(0, 0.3, sprintf('lambda = %d, gamma = %d\n', lambda, gamma));
    text(0, 0.0, sprintf('data file %s\n', strrep(file, '_', '\_')));
    set(ax, 'visible', 'off');
    scatter(subplot(3,2,6),Xt(:,1),Xt(:,2),25,sign(predictions),'filled'); title('Rounded Predictions');
else
    fprintf('dimension %d is too large to plot\n', size(Xt, 2));
    fprintf('works for %d of %d inputs\n', sum(Yt - sign(predictions) == 0), size(Xt, 1));
end