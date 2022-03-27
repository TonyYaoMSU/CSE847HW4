function [a] = sigmoid(b)
    a = 1./(1+exp(-b));
end