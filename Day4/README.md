

## Transfer Learning Code Modification: ResNet-18 to VGG16

Model type change:
You load VGG-16 instead of ResNet-18 because it’s a different CNN architecture that uses sequential convolutional and fully connected layers.

Freezing layers:
You freeze only the feature extraction part (called features in VGG). That’s because the convolutional layers already learned good generic features from ImageNet, and we don’t want to retrain them.

Replacing final layer:
The last layer in VGG is inside a block called classifier, not fc.
You replace the last fully connected layer with one that outputs 2 classes (ants and bees).
This is done because the original model predicts 1000 ImageNet classes, not 2.

## Transfer Learning Code Modification: ResNet-18 to inceptionv3
Model type change:
You load Inception V3, which is a deeper and more complex architecture using multiple convolution paths of different sizes.

Input size difference:
Inception expects images of size 299×299, not 224×224 like ResNet or VGG, so your image transformations must resize and crop to that size.

Freezing layers:
You freeze all pretrained parameters so that the network retains the knowledge learned from ImageNet and you only train the final layer for your two classes.

Replacing final layer:
Like ResNet, Inception has its final layer named fc, and you replace it with one having 2 outputs (for your custom dataset).

Auxiliary output:
Inception V3 produces two outputs — a main output and an auxiliary output used to help training.
During testing or evaluation, you only use the main output for predictions.

## Transfer Learning Code Modification: DenseNet121
When switching from ResNet-18 → DenseNet-121

Model type:
You load DenseNet-121, which connects each layer to every other layer within a block — this helps reuse features and reduce parameters.

Freezing layers:
You freeze all pretrained layers because DenseNet has already learned useful visual features from ImageNet.

Replacing final layer:
DenseNet’s final classifier is called classifier.
You replace it with a new linear layer for your 2 output classes (ants and bees).

Reason:
The original DenseNet classifier outputs 1000 ImageNet classes — we modify it to match our dataset.

## Transfer Learning Code Modification: MobileNetV2
Model type:
You load MobileNetV2, which is optimized for lightweight models — it’s efficient and ideal for mobile or edge devices.

Freezing layers:
You freeze all pretrained parameters because MobileNet’s base feature extractor is already trained on ImageNet.

Replacing final layer:
MobileNet stores its final classifier in a classifier block (like DenseNet).
You replace the last linear layer in that block to output 2 classes.

Reason:
The original classifier outputs 1000 ImageNet classes, so you change it to 2 for your dataset.
