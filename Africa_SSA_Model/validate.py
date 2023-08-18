import os
import torch
import json
import numpy as np
import nibabel as nib
from functools import partial

from monai import transforms
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from monai import data


def datafold_read(datalist, basedir, fold=0, key="validation"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k in d:
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    # tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        # else:
        #     tr.append(d)

    return val


def get_loader(data_dir, json_list, fold):
    
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            # transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image"])
        ]
    )

    validation_files = datafold_read(datalist=json_list, basedir=data_dir, fold=fold)
    val_ds = data.Dataset(data=validation_files, transform=val_transform)
    


    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    return val_loader


def main():

    # Set the dataset root directory and hyperparameters
    data_dir = "/scratch/guest185/"
    output_dir = "./Seg-images"
    roi = (128, 128, 128)     # Set the size to 240x240x155 each
    json_list = "./brats2023_ssa_val_data.json"
    # batch_size = 1     # changed from 2 to 1    
    # sw_batch_size = 2    
    fold = 0
    infer_overlap = 0.7   # changed from 0.5. to 0.7
    # max_epochs = 12       
    # val_every = 3 

    val_loader = get_loader(data_dir=data_dir, json_list=json_list, fold=fold)  

    # Creating Swin UNETR model
    model = SwinUNETR(
        img_size=roi,
        in_channels=4,
        out_channels=3,
        feature_size=60,    # changed from 48 to 60
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    )

    # Load the pre-trained weights
    model.load_state_dict(torch.load("./model_AdamW_CosineAnnealing_V2.pth"))

    # Set the model to evaluation mode
    model.eval()


    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[roi[0], roi[1], roi[2]],
        sw_batch_size=1,
        predictor=model,
        overlap=infer_overlap,
    )


    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            image = batch["image"].to(device)
            affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            num = '-'.join(batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("/")[-1].split("-")[2:4])
            img_name = "{}-seg.nii.gz".format(num)
            print("Inference on case {}".format(img_name))
            
            prob = torch.sigmoid(model_inferer_test(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]), dtype=np.uint8)
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 3
            
            nifti_header = nib.Nifti1Header()
            nifti_header.set_data_shape((240, 240, 155))
            nifti_header.set_xyzt_units('mm', 'sec')
            nifti_header.set_qform(np.diag([-1, 1, -1, 1]), code=1)
            nifti_header.set_sform(np.diag([-1, 1, -1, 1]), code=1)
            nifti_header['qoffset_x'] = 0
            nifti_header['qoffset_y'] = -239
            nifti_header['qoffset_z'] = 0
            
            nib.save(nib.Nifti1Image(seg_out, affine, header=nifti_header), os.path.join(output_dir, img_name))

        print("Finished inference!")



if __name__ == "__main__":

    main()
    