% Citire folder curent script MATLAB
currentFolder = 'C:\Users\Mihai\Desktop\FINAL_LICENTA_TOT\LICENTA2'

% Numele folderului cu date de antrenare
folderName = 'audio';

% Calea catre folderul cu date de antrenare
datasetLocation = fullfile(currentFolder, folderName);

% Verificare daca exista folderul
if ~exist(datasetLocation, 'dir')
    error('Dataset folder not found. Please make sure the dataset folder is available in the specified location.');
end

disp('Dataset folder found. Proceed with further operations as needed.');
%% accesarea folderului audio

ads = audioDatastore(currentFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
[adsTrain,adsValidation] = splitEachLabel(ads,0.8,0.2);

[x,fileInfo] = read(adsTrain);
fs = fileInfo.SampleRate;
reset(adsTrain)
sound(x,fs)
figure
t = (0:size(x,1)-1)/fs;
plot(t,x)
xlabel('Time (s)')
title('State = ' + string(fileInfo.Label))
axis tight

%% prelucrarea audio-urilor

emptyLabelVector = adsTrain.Labels;
emptyLabelVector(:) = [];

trainFeatures = [];
trainLabels = emptyLabelVector;
while hasdata(adsTrain)
    [audioIn,fileInfo] = read(adsTrain);
    features = yamnetPreprocess(audioIn,fileInfo.SampleRate);
    numSpectrums = size(features,4);
    trainFeatures = cat(4,trainFeatures,features);
    trainLabels = cat(2,trainLabels,repmat(fileInfo.Label,1,numSpectrums));
end

validationFeatures = [];
validationLabels = emptyLabelVector;
while hasdata(adsValidation)
    [audioIn,fileInfo] = read(adsValidation);
    features = yamnetPreprocess(audioIn,fileInfo.SampleRate);
    numSpectrums = size(features,4);
    validationFeatures = cat(4,validationFeatures,features);
    validationLabels = cat(2,validationLabels,repmat(fileInfo.Label,1,numSpectrums));
end

%% urmatoarea parte - adaugarea retelei YAMNet in mediul de lucru

currentFolder = 'C:\Users\Mihai\Desktop\FINAL_LICENTA_TOT\LICENTA2';

YAMNetFolder = 'yamnet';

YAMNetLocation = fullfile(currentFolder, YAMNetFolder);

if ~exist(YAMNetLocation, 'dir')
    error('YAMNet folder not found. Please make sure the YAMNet folder is available in the specified location.');
end

addpath(YAMNetLocation);

disp('YAMNet adaugat in mediul MATLAB.');

%% urmatoarea parte - modificarea ultimelor 2 straturi in retea

uniqueLabels = unique(adsTrain.Labels);
numLabels = numel(uniqueLabels);

net = yamnet;

lgraph = layerGraph(net.Layers);

newDenseLayer = fullyConnectedLayer(numLabels,"Name","dense");
lgraph = replaceLayer(lgraph,"dense",newDenseLayer);

newClassificationLayer = classificationLayer("Name","Sounds","Classes",uniqueLabels);
lgraph = replaceLayer(lgraph,"Sound",newClassificationLayer);

%% parametrii antrenare

miniBatchSize = 32;
validationFrequency = floor(numel(trainLabels) / miniBatchSize / 2);

options = trainingOptions('adam', ...
    'MaxEpochs',50, ...  
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'InitialLearnRate',0.00005, ... 
    'L2Regularization',0.005,... 
    'ValidationData',{single(validationFeatures),validationLabels}, ...
    'ValidationFrequency',validationFrequency,...
    'ExecutionEnvironment','gpu', ...
    'ValidationPatience',5);


ymnt = trainNetwork(trainFeatures,trainLabels,lgraph,options);
save ymnt.mat ymnt

%% validarea rezultatelor

load('ymnt.mat');

predictedLabels = classify(ymnt, single(validationFeatures));

C = confusionmat(validationLabels, predictedLabels);

figure;
confusionchart(C, unique(validationLabels), 'RowSummary','row-normalized', 'ColumnSummary','column-normalized');
title('Confusion Matrix');
