function [cost,grad] = sparseAutoencoderCostVectorized(theta, ...
    visibleSize, hiddenSize, lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% %% FOR-LOOP BASED
% 
% % calculate sparsity derivative: feedforward
% m = size(data,2);
% z2 = W1 * data + repmat(b1,1,m);
% a2 = f(z2);
% z3 = W2 * a2 + repmat(b2,1,m);
% h = f(z3);
% 
% rho_hat = sum(a2,2)/m;
% sparsity_deriv = beta*...
%     (-sparsityParam./rho_hat + (1-sparsityParam)./(1-rho_hat));
% 
% % cost due to error
% err = h - data;
% cost_err = sum(sum(err.*err))/2;
% 
% % deltas
% delta3 = err.*fprime(h);
% delta2 = (W2'*delta3 + repmat(sparsity_deriv,1,m)).*fprime(a2);
% 
% % gradients
% W2grad = delta3*a2'/m + lambda*W2;
% W1grad = delta2*data'/m + lambda*W1;
% b2grad = mean(delta3,2);
% b1grad = mean(delta2,2);
% 
% KLdiv = sparsityParam*log(sparsityParam./rho_hat) + ...
%     (1 - sparsityParam)*log((1 - sparsityParam)./(1 - rho_hat));
% 
% cost_err = cost_err/m;
% cost_weights = lambda/2*(sum(W1(:).^2) + sum(W2(:).^2)); % w regularization
% cost_sparse = beta*sum(KLdiv); % induce "sparsity"
% 
% cost = cost_err + cost_weights + cost_sparse;

%% VECTORIZED

% calculate sparsity derivative: feedforward
m = size(data,2);
z2 = W1 * data + repmat(b1,1,m);
a2 = f(z2);
z3 = W2 * a2 + repmat(b2,1,m);
h = f(z3);

rho_hat = sum(a2,2)/m;
sparsity_deriv = beta*...
    (-sparsityParam./rho_hat + (1-sparsityParam)./(1-rho_hat));

% cost due to error
err = h - data;
cost_err = sum(sum(err.*err))/2;

% deltas
delta3 = err.*fprime(h);
delta2 = (W2'*delta3 + repmat(sparsity_deriv,1,m)).*fprime(a2);

% gradients
W2grad = delta3*a2'/m + lambda*W2;
W1grad = delta2*data'/m + lambda*W1;
b2grad = mean(delta3,2);
b1grad = mean(delta2,2);

KLdiv = sparsityParam*log(sparsityParam./rho_hat) + ...
    (1 - sparsityParam)*log((1 - sparsityParam)./(1 - rho_hat));

cost_err = cost_err/m;
cost_weights = lambda/2*(sum(W1(:).^2) + sum(W2(:).^2)); % w regularization
cost_sparse = beta*sum(KLdiv); % induce "sparsity"

cost = cost_err + cost_weights + cost_sparse;

%%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

function h = f(z)
    h = sigmoid(z);
end

function fpr = fprime(a)
    fpr = a.*(1-a);
end
