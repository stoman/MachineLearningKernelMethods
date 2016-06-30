%This function predicts the classes of some objects using a given
%prediction function. predict is the prediction function and X and Y are
%the test data. The return value is the number of correct estimates.
%Author: Stefan Toman (toman@tum.de)
function correct = predictionquality(predict, X, Y)
    predictions = predict(X);
    correct = sum(Y - sign(predictions) == 0);
end
