clear
format short
format compact

X = 1

imdsT = imageDatastore('training', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
labelCountT = countEachLabel(imdsT)

imdsV = imageDatastore('testing', ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
labelCountV = countEachLabel(imdsV)

imdsTrain = imdsT
imdsValidation = imdsV


% digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
%     'nndatasets','DigitDataset');
% imds = imageDatastore(digitDatasetPath, ...
%     'IncludeSubfolders',true,'LabelSource','foldernames');
% numTrainFiles = 750;
% [imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

% modifica a passada dos filtro da camadas conv.
for fS = 1:5
    % modifica o tamanho dos filtros das camadas conv.
    for fT = 1:5
        % apenas testar quano a passada for menor ou igual ao filtro
        if fS <= fT 
            % modifica a quanidade de filtros por camada conv.
            for fQ = [1,2,4,8]
               
                            
                % inicio cronometro
                tI = now;
                
                %arruma o padding manualmente
                padding = fS-1;                               
                
                %arquivo log
                arquivo = ['resultadosFinais/resultado_' num2str(fS,'%d') '_' num2str(fT,'%d') '_' num2str(fQ*2,'%d') '-' num2str(fQ*8,'%d') '.txt'];
                diary(arquivo);
                diary on;
                disp(['fT = ' num2str(fT,'%d') ' fS = ' num2str(fS,'%d') ' fQ = '  num2str(fQ,'%d') ' ('  num2str(fQ*2,'%d') '/'  num2str(fQ*8,'%d') ') padding = '  num2str(padding,'%d')])
                %configuracoes de treinamento

                layers = [
                    imageInputLayer([28 28 1])

                    convolution2dLayer(fT,fQ*2,'Stride',fS)
                    batchNormalizationLayer
                    reluLayer

                    maxPooling2dLayer(2,'Stride',2)

                    convolution2dLayer(fT,fQ*8,'Stride',fS,'Padding',padding)
                    batchNormalizationLayer
                    reluLayer

                    fullyConnectedLayer(10)
                    softmaxLayer
                    classificationLayer];

                options = trainingOptions('sgdm', ...
                    'VerboseFrequency', 250,... 
                    'MaxEpochs',15,...
                    'ValidationData',imdsValidation, ...
                    'ValidationPatience',5, ...
                    'ValidationFrequency',250);
                
                % treina a CNN
                net = trainNetwork(imdsTrain,layers,options);

                % testa a CNN
                YPred = classify(net,imdsValidation);
                YValidation = imdsValidation.Labels;        
                accuracy(fT,fS,fQ) = sum(YPred == YValidation)/numel(YValidation);
                
                % Acuracia e Tempo total ate o momento
                disp(['Accu. = ' num2str(accuracy(fT,fS,fQ),'%f')])
                
                % fim cronometro
                tF = now;
                
                tempoTotal(X) = tF - tI;
                tempoMedio = mean(tempoTotal);
                disp(['Tempo = ' datestr(tempoTotal(X),'HH:MM:SS.FFF')])
                disp(['Tempo total = ' datestr(sum(tempoTotal),'HH:MM:SS.FFF')])
                %salva o log
                diary off;
                clc
                tempoResta = tempoMedio * (60-X);
                X = X+1;
                disp(['Restante = ' num2str(60-X,'%d') ' (aprox. ' datestr(tempoResta,'HH:MM:SS') ')'])
                
            end
        end
    end
end
