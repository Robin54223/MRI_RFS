from monai.transforms import (
    Compose,
    ToTensord,
    RandFlipd,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    AddChanneld,
    DivisiblePadd
)


#Transforms to be applied on training instances
train_transform = Compose(
    [
        AddChanneld(keys=["image1", "image2"]),
        #Spacingd(keys=["image1", "image2", 'label'], pixdim=(1., 1., 1.), mode=("bilinear","bilinear","bilinear","bilinear", "nearest")),
        # RandFlipd(keys=["image1", "image2"], prob=0.1, spatial_axis=0),
        # RandFlipd(keys=["image1", "image2"], prob=0.1, spatial_axis=1),
        # RandFlipd(keys=["image1", "image2"], prob=0.1, spatial_axis=2),
        # NormalizeIntensityd(keys=["image1", "image2"], nonzero=True, channel_wise=True),
        # RandScaleIntensityd(keys=["image1", "image2"], factors=0.1, prob=0.1),
        # RandShiftIntensityd(keys=["image1", "image2"], offsets=0.1, prob=0.1),
        DivisiblePadd(k=16, keys=["image1", "image2"]),
        ToTensord(keys=["image1", "image2", 'label_cls'])
    ]
)

#Cuda version of "train_transform"
train_transform_cuda = Compose(
    [
        AddChanneld(keys=["image1", "image2"]),
        #Spacingd(keys=["image1", "image2", 'label'], pixdim=(1., 1., 1.), mode=("bilinear","bilinear","bilinear","bilinear", "nearest")),
        # RandFlipd(keys=["image1", "image2"], prob=0.1, spatial_axis=0),
        # RandFlipd(keys=["image1", "image2"], prob=0.1, spatial_axis=1),
        # RandFlipd(keys=["image1", "image2"], prob=0.1, spatial_axis=2),
        # NormalizeIntensityd(keys=["image1", "image2"], nonzero=True, channel_wise=True),
        # RandScaleIntensityd(keys=["image1", "image2"], factors=0.1, prob=0.1),
        # RandShiftIntensityd(keys=["image1", "image2"], offsets=0.1, prob=0.1),
        DivisiblePadd(k=16, keys=["image1", "image2"]),
        ToTensord(keys=["image1", "image2", 'label_cls'], device='cuda')
    ]
)

#Transforms to be applied on validation instances
val_transform = Compose(
    [
        AddChanneld(keys=["image1", "image2"]),
        # NormalizeIntensityd(keys=["image1", "image2"], nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image1", "image2"]),
        ToTensord(keys=["image1", "image2", 'label_cls'])
    ]
)

#Cuda version of "val_transform"
val_transform_cuda = Compose(
    [
        AddChanneld(keys=["image1", "image2"]),
        # NormalizeIntensityd(keys=["image1", "image2"], nonzero=True, channel_wise=True),
        DivisiblePadd(k=16, keys=["image1", "image2"]),
        ToTensord(keys=["image1", "image2", 'label_cls'])
    ]
)