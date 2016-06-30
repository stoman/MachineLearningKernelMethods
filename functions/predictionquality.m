%This function predicts the classes of some objects using kernel methods. X
%and Y are the training data, lambda is the regularization parameter, K is
%the kernel function and Xt and Yt are the test data. The return value is
%the number of correct estimates.
%Author: Stefan Toman (toman@tum.de)
function correct = predictionquality(X, Y, lambda, K, Xt, Yt)
    predict = funpredict(X, Y, lambda, K);
    predictions = predict(Xt);
    correct = sum(Yt - sign(predictions) == 0);
end
