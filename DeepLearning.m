% The demostration of Deep Learning for the course of streaming Technology
% jzchen@hust.edu.cn
clc
clear
x=[0.05 0.10]';

W0=[0.15 0.20;
    0.25 0.30]; % 第零层的权值

W1=[0.15 0.20;
    0.25 0.30]; % 第一层的权值

W2=[0.40 0.45;
    0.50 0.55]; % 第二层的权值

B0=0.35;
B1=0.35; % 偏置
B2=0.60;

Eta=0.5; % 学习率

L=[0.01 0.99]'; % 初始标签

iteration=10000;

%保留每次迭代的误差
Etotal=zeros(1,iteration);

for i=1:iteration
    % 第0层推理    
    Y0=W0*x+B0;%
    A0=1./(1+exp(-Y0));%激活
    % 第1层推理    
    Y1=W1*A0+B1;%
    A1=1./(1+exp(-Y1));%激活
    % 第2层推理    
    Y2=W2*A1+B2;
    A2=1./(1+exp(-Y2));
    
    % 第2层学习
    E1=0.5*(L(1)-A2(1))^2; % Square error
    E2=0.5*(L(2)-A2(2))^2;
%     E1=L(1)*log2(A2(1)+eps); % Cross entropy
%     E2=L(2)*log2(A2(2)+eps);
    Etotal(i)=E1+E2;
    dEtotal_dA2=[-(L(1)-A2(1)) -(L(2)-A2(2))]'; % derivative of Square error
%     dEtotal_dA2=1/log(2)*[L(1)/(A2(1)+eps) L(2)/(A2(2)+eps)]'; % derivative of Cross entropy
    dA2_dY2=[A2(1)*(1-A2(1)) A2(2)*(1-A2(2))]'; % 默认为Sigmod函数
    dY2_dW2=[A1(1) A1(2);A1(1) A1(2)];
    Delta_2=[dEtotal_dA2(1)*dA2_dY2(1) dEtotal_dA2(2)*dA2_dY2(2)];
    dEtotal_dW2=[Delta_2(1)*dY2_dW2(1,1) Delta_2(1)*dY2_dW2(1,2);Delta_2(2)*dY2_dW2(2,1) Delta_2(2)*dY2_dW2(2,2)];    
    % 第1层学习
    dY2_dA1=[W2(1,1) W2(1,2); W2(2,1) W2(2,2)]; % W2到底在之前还是之后更新呢?
    dEtotal_dY2=[dEtotal_dA2(1)*dA2_dY2(1) dEtotal_dA2(2)*dA2_dY2(2)]; % 这两句话可用这一句替代dEtotal_dA1=[Delta_2*dY2_dA1(:,1) Delta_2*dY2_dA1(:,2)];
    dEtotal_dA1=[dEtotal_dY2*dY2_dA1(:,1) dEtotal_dY2*dY2_dA1(:,2)];
    dA1_dY1=[A1(1)*(1-A1(1)) A1(2)*(1-A1(2))];
    dY1_dW1=[A0(1) A0(2);A0(1) A0(2)];
    Delta_1=[dEtotal_dA1(1)*dA1_dY1(1) dEtotal_dA1(2)*dA1_dY1(2)];
    dEtotal_dW1=[Delta_1(1)*dY1_dW1(1,1) Delta_1(1)*dY1_dW1(1,2);Delta_1(2)*dY1_dW1(2,1) Delta_1(2)*dY1_dW1(2,2)];
    
    % 第0层学习
    dY1_dA0=[W1(1,1) W1(1,2); W1(2,1) W1(2,2)]; % W2到底在之前还是之后更新呢?
    dEtotal_dY1=[dEtotal_dA1(1)*dA1_dY1(1) dEtotal_dA1(2)*dA1_dY1(2)]; % 这两句话可用这一句替代dEtotal_dA1=[Delta_2*dY2_dA1(:,1) Delta_2*dY2_dA1(:,2)];
    dEtotal_dA0=[dEtotal_dY1*dY1_dA0(:,1) dEtotal_dY1*dY1_dA0(:,2)];
    dA0_dY0=[A0(1)*(1-A0(1)) A0(2)*(1-A0(2))];
    dY0_dW0=[x(1) x(2);x(1) x(2)];
    Delta_0=[dEtotal_dA0(1)*dA0_dY0(1) dEtotal_dA0(2)*dA0_dY0(2)];
    dEtotal_dW0=[Delta_0(1)*dY0_dW0(1,1) Delta_0(1)*dY0_dW0(1,2);Delta_0(2)*dY0_dW0(2,1) Delta_0(2)*dY0_dW0(2,2)];
    
    % 权值更新
    W2=W2-Eta*dEtotal_dW2; % 第2层
    W1=W1-Eta*dEtotal_dW1; % 第1层
    W0=W0-Eta*dEtotal_dW0; % 第0层
end
disp(['Labels: ',num2str(L') ', Predicitons: ',num2str(A2') ', Prediciton Errors: ',num2str(Etotal(iteration))])
plot(Etotal,'LineWidth',3.5)
grid on;
ylabel('Prediction Error','FontSize',20);
xlabel('Iterations','FontSize',20);