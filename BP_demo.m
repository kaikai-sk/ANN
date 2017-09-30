% A demo to BP nerual network

format long
% define the sigmoid function
f = @(x) 1/(1+exp(-x));
% enter the learning rate
Eta = 0.5;

x=[0.05,0.10];
y=[0.01,0.99];

b=[1,1];
bw=[0.35,0.6];

temp = size(x);
 % number of rows
m = temp(1);
% number of cols
n = temp(2);

% the weights
w=[0.15,0.20,0.25,0.30;
    0.40,0.45,0.50,0.55];
% gradient 
gradient = zeros(2,4);

% the output of neurons in hidden layer
hout = [1;1];
% the output of neurons in hidden layer
oout=[1;1];

    
% initialize the condition to terminate loop
diff = 1;


while(diff <= 1)
    %for k = 1:m  % for all samples
        
        % the obtained output
        neth1=w(1,1)*x(1)+w(1,2)*x(2) + b(1)*bw(1);
        hout(1) = f( neth1 );
        hout(2) = f( w(1,3)*x(1)+w(1,4)*x(2) + b(1)*bw(1) );
        oout(1)=f(  w(2,1)*hout(1)+w(2,2)*hout(2) + b(2)*bw(2) );
        oout(2)=f(  w(2,3)*hout(1)+w(2,4)*hout(2) + b(2)*bw(2) );
        % gradient of output layer
        deltaO=[1;1];
        deltaO(1)=( oout(1) - y(1) ) * oout(1) * (1 - oout(1) );
        deltaO(2)=( oout(2) - y(2) ) * oout(2) * (1 - oout(2) );
        
        %求隐藏层到输出层的偏导
        gradient(2,1)=deltaO(1)*hout(1);
        gradient(2,2)=deltaO(1)*hout(2);
        gradient(2,3)=deltaO(2)*hout(1);
        gradient(2,4)=deltaO(2)*hout(2);
        
        gradient(1,:);
        %w(2,:)   
        %求输入层到隐藏层的偏导
        gradient(1,1)=( deltaO(1)*w(2,1) + deltaO(2)*w(2,3) )*hout(1)*(1-hout(1))*x(1);
        gradient(1,2)=( deltaO(1)*w(2,1) + deltaO(2)*w(2,3) )*hout(1)*(1-hout(1))*x(2);
        gradient(1,3)=( deltaO(2)*w(2,4) + deltaO(1)*w(2,2) )*hout(2)*(1-hout(2))*x(1);
        gradient(1,4)=( deltaO(2)*w(2,4) + deltaO(1)*w(2,2) )*hout(2)*(1-hout(2))*x(2);
        
%         gradient(1,:);
        
        %进行梯度下降
        w=w-Eta*gradient
        
    % update diff
    diff=diff+1;
end