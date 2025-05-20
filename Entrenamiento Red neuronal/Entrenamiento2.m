clc; close all; clear all;

#Cargar datos entrenamiento
bien=csvread('train_bien.csv');
mal=csvread('train_mal.csv');

F1=0;
while (F1<92)
%%c=c+1
% dentro del parentesis los primeros : es para
%tomar todos los valores en renglones
%despues de la , es para tomar los valores en columnas
%Variable para determinar el tamaño de nuestros vectores
Dt=1200
Bien=bien(1:Dt,1:33);
Mal=mal(1:Dt,1:33);

##################################
function A = softmax(Z)
expZ=exp(Z-max(Z,[],1));
A=expZ ./sum(expZ,1);
end
%%%%%%Normalizacion de datos de entrenamiento
datos=[Bien;Mal];
ValMin=min(datos);
ValMax=max(datos);
ymax=1;
ymin=0.1;

Dato_norm=((ymax-ymin)*(datos-ValMin))./(ValMax-ValMin)+ymin;
% Entradas para la red neuronal
%%input = [Bien', Mal'];
input=[Dato_norm'];

% Targets (salidas deseadas)

T_Bien = [ones(1, Dt); zeros(1, Dt)];
T_Mal = [zeros(1, Dt); ones(1, Dt)];
targets = [T_Bien, T_Mal];

% Posible un ciclo while para entrenar hasta llegar al 80% o superior
% Hiperparámetros

%%if (c>10)
%%c=0;
%%numhiden=numhidden+10;
num_inputs = size(input, 1);
num_hidden = 100; % Número de neuronas en la capa oculta
num_hidden2 = 75;
num_outputs = size(targets, 1);
learning_rate = 0.001;
epochs = 1000;

% Inicialización de pesos y sesgos
W1 = randn(num_hidden, num_inputs);
W2 = randn(num_hidden2, num_hidden);
W3 = randn(num_outputs, num_hidden2);

Init_W1=W1;
Init_W2=W2;
Init_W3=W3;
%W1 = randn(num_hidden, num_inputs) * 0.1; % Pesos de entrada -> capa oculta
b1 = randn(num_hidden, 1) * 0.1; % Sesgos de la capa oculta
%W2 = randn(num_outputs, num_hidden) * 0.1; % Pesos capa oculta -> salida
b2 = randn(num_hidden2, 1) * 0.1; % Sesgos de la capa de salida
b3 = randn(num_outputs, 1) * 0.1;

Init_b1=b1;
Init_b2=b2;
Init_b3=b3;
% Entrenamiento de la red neuronal
for epoch = 1:epochs
% Propagación hacia adelante
  Z1 = W1 * input + b1;
  A1 = tanh(Z1);
  Z2 = W2 * A1 + b2;
  A2 = tanh(Z2);
  Z3 = W3 * A2 + b3;
  A3 = softmax(Z3);

% Cálculo del error (entropía cruzada)
loss = -sum(sum(targets .* log(A3))) / size(targets, 2);

% Propagación hacia atrás (backpropagation)
  dZ3 = A3 - targets;
  dW3 = dZ3 * A2';
  db3 = sum(dZ3, 2);

  dA2 = W3' * dZ3;
  dZ2 = dA2 .* (1 - A2.^2);
  dW2 = dZ2 * A1';
  db2 = sum(dZ2, 2);

  dA1 = W2' * dZ2;
  dZ1 = dA1 .* (1 - A1.^2);
  dW1 = dZ1 * input';
  db1 = sum(dZ1, 2);

% Actualización de pesos y sesgos
lambda = 0.05; % Factor de regularización
W1 = W1 - learning_rate * (dW1 + lambda * W1);
W2 = W2 - learning_rate * (dW2 + lambda * W2);
W3 = W3 - learning_rate * (dW3 + lambda * W3);


%W1 = W1 - learning_rate * dW1;
b1 = b1 - learning_rate * db1;
%W2 = W2 - learning_rate * dW2;
b2 = b2 - learning_rate * db2;
b3 = b3 - learning_rate * db3;

% Mostrar la pérdida cada 100 épocas
if mod(epoch, 100) == 0
fprintf('Epoch %d, Loss: %.4f\n', epoch, loss);
end
end

% Evaluación de la red
Z1 = W1 * input + b1;
A1 = tanh(Z1);
Z2 = W2 * A1 + b2;
A2 = tanh(Z2);
Z3 = W3 * A2 + b3;
A3 = softmax(Z3);

% Predicciones
[~, y_pred] = max(A3  , [], 1); % Clases predichas
[~, y_test] = max(targets, [], 1); % Clases reales

% Matriz de confusión
confMat = zeros(num_outputs, num_outputs);
for i = 1:length(y_test)
confMat(y_test(i), y_pred(i)) = confMat(y_test(i), y_pred(i)) + 1;
end
disp('Matriz de confusión:');
disp(confMat);

% Cálculo de métricas
accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;
disp(['Precisión: ', num2str(accuracy), '%']);
disp(['Datos de entrenamiento']);
% Red creada
red.W1 = W1;
red.b1 = b1;
red.W2 = W2;
red.b2 = b2;
red.W3 = W3;
red.b3 = b3;


%%%Carga de datos de evaluacion

Test_bien=csvread('test_bien.csv');
Test_mal=csvread('test_mal.csv');

% dentro del parentesis los primeros : es para
%tomar todos los valores en renglones
%despues de la , es para tomar los valores en columnas
%Variable para determinar el tamaño de nuestros vectores

Test_Bien=Test_bien(:,1:33);
Test_Mal=Test_mal(:,1:33);

%%Normalizacion datos de prueba

%datos=[Bien;Mal];
%ValMin=min(datos);
%ValMax=max(datos);

ymax=1;
ymin=0.1;

Norm_bien_test=((ymax-ymin)*(Test_Bien-ValMin))./(ValMax-ValMin)+ymin;
Norm_mal_test=((ymax-ymin)*(Test_Mal-ValMin))./(ValMax-ValMin)+ymin;

%Metricas de evaluacion

TP=0;
FP=0;
TN=0;
FN=0;

%Evaluacion de la clase Bien
for i = 1:size(Norm_bien_test, 1)
  Z1 = red.W1 * Norm_bien_test(i,:)' + red.b1;
  A1 = tanh(Z1);
  Z2 = red.W2 * A1 + red.b2;
  A2 = tanh(Z2);
  Z3 = red.W3 * A2 + red.b3;
  A3 = softmax(Z3);

  if A3(1) > A3(2)
    TP = TP + 1;
  else
    FN = FN + 1;
  end
end

%Evaluacion de la clase Mal
for i = 1:size(Norm_mal_test, 1)
  Z1 = red.W1 * Norm_mal_test(i,:)' + red.b1;
  A1 = tanh(Z1);
  Z2 = red.W2 * A1 + red.b2;
  A2 = tanh(Z2);
  Z3 = red.W3 * A2 + red.b3;
  A3 = softmax(Z3);

  if A3(2) > A3(1)
    TN = TN + 1;
  else
    FP = FP + 1;
  end
end

Precision=(TP/(FP+TP))*100
Exactitud=((TP+TN)/(TN+TP+FP+FN))*100
Recall=(TP/(TP+FN))*100
F1=2*((Precision*Recall)/(Precision+Recall))
  if(F1>95)
        save('best_network.mat', 'red', '-v7');
        save('train_min_max.mat', 'ValMin', 'ValMax', 'ymin', 'ymax', '-v7');
  end
end

