from albumentations import *

augment0=Compose([HorizontalFlip()],p=1)
augment1=Compose([HorizontalFlip(),HueSaturationValue(p=0.5),RandomBrightnessContrast(p=0.5)],p=1)
augment_rand1=Compose([RandomCrop(380,380),HorizontalFlip(),HueSaturationValue(p=0.5),RandomBrightnessContrast(p=0.5)],p=1)
augment2=Compose([HorizontalFlip(),HueSaturationValue(p=0.5),RandomBrightnessContrast(p=0.5),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.3),
        OneOf([
            MotionBlur(),
            GaussianBlur(),
            JpegCompression(quality_lower=65, quality_upper=80),
        ], p=0.3),ToGray(p=0.1)],p=1)

augmentations={'augment0':augment0,'augment1':augment1,'augment2':augment2,'augment_rand1':augment_rand1}