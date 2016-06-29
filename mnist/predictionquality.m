function correct = predictionquality(X, Y, lambda, K, Xt, Yt)
    predict = funpredict(X, Y, lambda, K);
    predictions = predict(Xt);
    correct = sum(Yt - sign(predictions) == 0);
end
