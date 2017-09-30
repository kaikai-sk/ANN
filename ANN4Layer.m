% A demo to BP nerual network

format long
% define the sigmoid function
f = @(x) 1/(1+exp(-x));
% enter the learning rate
Eta = 0.5;

x=[0.05,0.10];
y=[0.01,0.99];

% modified
b=[1,1,1];
bw=[0.35,0.6,0.35];

temp = size(x);
 % number of rows
m = temp(1);
% number of cols
n = temp(2);

% the weights
w=[0.15,0.20,0.25,0.30;
    0.40,0.45,0.50,0.55;
    0.40,0.45,0.50,0.55];
% gradient 
gradient = zeros(3,4);

% the output of neurons in hidden layer
hout = zeros(2,2);
% the output of neurons in hidden layer
oout=[1;1];

    
% initialize the condition to terminate loop
diff = 1;


while(diff <= 1)
        % �������������->���ز�1��
        neth1=w(1,1)*x(1)+w(1,2)*x(2) + b(1)*bw(1);
        hout(1,1) = f( neth1 );
        hout(1,2) = f( w(1,3)*x(1)+w(1,4)*x(2) + b(1)*bw(1) );

        %(���ز�1->���ز�2)
        hout(2,1)=f( w(2,1)*hout(1,1)+w(2,2)*hout(1,2) + b(2)*bw(2) )
        hout(2,2)=f( w(2,3)*hout(1,1)+w(2,4)*hout(1,2) + b(2)*bw(2) )
        
        %(���ز�2->�����)
        oout(1)=f(  w(3,1)*hout(2,1)+w(3,2)*hout(2,2) + b(3)*bw(3) );
        oout(2)=f(  w(3,3)*hout(2,1)+w(3,4)*hout(2,2) + b(3)*bw(3) );
       
        % gradient of output layer
        deltaO=zeros(2,4);
        deltaO(1,4)=( oout(1) - y(1) ) * oout(1) * (1 - oout(1) );
        deltaO(2,4)=( oout(2) - y(2) ) * oout(2) * (1 - oout(2) );
        
        %������㵽���ز�2��ƫ��
        gradient(3,1)=deltaO(1,4)*hout(2,1);
        gradient(3,2)=deltaO(1,4)*hout(2,2);
        gradient(3,3)=deltaO(2,4)*hout(2,1);
        gradient(3,4)=deltaO(2,4)*hout(2,2);
        
        %��deltaO (3)
        deltaO(1,3) = ( deltaO(1,4)*w(2,1) + deltaO(2,4)*w(2,3) )*hout(2,1)*(1-hout(2,1));
        deltaO(2,3) = ( deltaO(2,4)*w(2,4) + deltaO(1,4)*w(2,2) )*hout(2,2)*(1-hout(2,2));
        
        gradient(1,:);
        %w(2,:)   
        
        %�����ز�2�����ز�1��ƫ��
        gradient(2,1)= deltaO(1,3) * hout(1,1);
        gradient(2,2)= deltaO(1,3) * hout(1,2);
        gradient(2,3)= deltaO(2,3) * hout(1,1);
        gradient(2,4)= deltaO(2,3) * hout(1,2);
        
        %��deltaO (2)
        deltaO(1,2) = ( deltaO(1,4)*w(2,1) + deltaO(2,4)*w(2,3) )*hout(1,1)*(1-hout(1,1));
        deltaO(2,2) = ( deltaO(2,4)*w(2,4) + deltaO(1,4)*w(2,2) )*hout(1,2)*(1-hout(1,2));
        
        %�����ز�1��������ƫ��
        gradient(1,1)= deltaO(1,2) * x(1);
        gradient(1,2)= deltaO(1,2) * x(2);
        gradient(1,3)= deltaO(2,2) * x(1);
        gradient(1,4)= deltaO(2,2) * x(2);
        
%         gradient(1,:);
        
        %�����ݶ��½�
        w=w-Eta*gradient
        
    % update diff
    diff=diff+1;
end