# ------------------------------------------------------------------
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# ------------------------------------------------------------------
# Modified by Yiming Zhou
# ------------------------------------------------------------------

import logging
import os
import sys
import click
import numpy as np
import torch
sys.path.append("src")
import backbones
import common
import metrics
import simplenet 
import utils

LOGGER = logging.getLogger(__name__)
_DATASETS = {
    "mvtec": ["datasets.mvtec", "MVTecDataset"],
}


@click.group(chain=True)
@click.option("--results_path", type=str, default='results')
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="simplenet_mvtec")
@click.option("--log_project", type=str, default="MVTecAD_Results")
@click.option("--run_name", type=str, default="run")
@click.option("--test", is_flag=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    run_name,
    test,
):
    print("!"*100)
    print(f'methods:{methods}\n\
            results_path:{results_path}\n\
            gpu:{gpu}\n\
            seed:{seed}\n\
            log_group:{log_group}\n\
            log_project:{log_project}\n\
            run_name:{run_name}\n\
            test:{test}\n\
                ')
    
    methods = {key: item for (key, item) in methods}
    

    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    pid = os.getpid()
    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = utils.set_torch_device(gpu)

    result_collect = []
    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name
        if dataset_name != 'mvtec_pill':
            continue

        imagesize = dataloaders["training"].dataset.imagesize
        simplenet_list = methods["get_simplenet"](imagesize, device)

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        for i, SimpleNet in enumerate(simplenet_list):
            # torch.cuda.empty_cache()
            if SimpleNet.backbone.seed is not None:
                utils.fix_seeds(SimpleNet.backbone.seed, device)
            LOGGER.info(
                "Training models ({}/{})".format(i + 1, len(simplenet_list))
            )
            # torch.cuda.empty_cache()

            SimpleNet.set_model_dir(os.path.join(models_dir, f"{i}"), dataset_name)
            if not test:
                i_auroc, p_auroc, pro_auroc = SimpleNet.train(dataloaders["training"], dataloaders["testing"])
            else:
                i_auroc, p_auroc, pro_auroc =  SimpleNet.test(dataloaders["training"], dataloaders["testing"])


            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": i_auroc, # auroc,
                    "full_pixel_auroc": p_auroc, # full_pixel_auroc,
                    "anomaly_pixel_auroc": pro_auroc,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )


# @main.command("net")
# @click.option("--backbone_names", "-b", type=str, multiple=True, default=['wideresnet50'])
# @click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=['layer2','layer3'])
# @click.option("--pretrain_embed_dimension", type=int, default=1536)
# @click.option("--target_embed_dimension", type=int, default=1536)
# @click.option("--patchsize", type=int, default=3)
# @click.option("--embedding_size", type=int, default=256)
# @click.option("--meta_epochs", type=int, default=200)
# @click.option("--aed_meta_epochs", type=int, default=1)
# @click.option("--gan_epochs", type=int, default=4)
# @click.option("--dsc_layers", type=int, default=2)
# @click.option("--dsc_hidden", type=int, default=1024)
# @click.option("--noise_std", type=float, default=0.015)
# @click.option("--dsc_margin", type=float, default=0.5)
# @click.option("--dsc_lr", type=float, default=0.0002)
# @click.option("--auto_noise", type=float, default=0)
# @click.option("--train_backbone", is_flag=True)
# @click.option("--cos_lr", is_flag=True)
# @click.option("--pre_proj", type=int, default=1)
# @click.option("--proj_layer_type", type=int, default=0)
# @click.option("--mix_noise", type=int, default=1)
def net(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    patchsize,
    embedding_size,
    meta_epochs,
    aed_meta_epochs,
    gan_epochs,
    noise_std,
    dsc_layers, 
    dsc_hidden,
    dsc_margin,
    dsc_lr,
    auto_noise,
    train_backbone,
    cos_lr,
    pre_proj,
    proj_layer_type,
    mix_noise,
):
    print("$"*100)
    print(f"backbone_names:{backbone_names},\n\
            layers_to_extract_from:{layers_to_extract_from}, \n\
            pretrain_embed_dimension:{pretrain_embed_dimension},\n\
            target_embed_dimension:{target_embed_dimension},\n\
            patchsize:{patchsize},\n\
            embedding_size:{embedding_size},\n\
            meta_epochs:{meta_epochs},\n\
            aed_meta_epochs:{aed_meta_epochs},\n\
            gan_epochs:{gan_epochs},\n\
            noise_std:{noise_std},\n\
            dsc_layers:{dsc_layers},\n\
            dsc_hidden:{dsc_hidden},\n\
            dsc_margin:{dsc_margin},\n\
            dsc_lr:{dsc_lr},\n\
            auto_noise:{auto_noise},\n\
            train_backbone:{train_backbone},\n\
            cos_lr:{cos_lr},\n\
            pre_proj:{pre_proj},\n\
            proj_layer_type:{proj_layer_type},\n\
            mix_noise:{mix_noise}")
    
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_simplenet(input_shape, device):
        simplenets = []
        for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = backbones.load(backbone_name)#获取主干网，此处为resnet50
            backbone.name, backbone.seed = backbone_name, backbone_seed

            simplenet_inst = simplenet.SimpleNet(device)
            simplenet_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                embedding_size=embedding_size,
                meta_epochs=meta_epochs,
                aed_meta_epochs=aed_meta_epochs,
                gan_epochs=gan_epochs,
                noise_std=noise_std,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                dsc_lr=dsc_lr,
                auto_noise=auto_noise,
                train_backbone=train_backbone,
                cos_lr=cos_lr,
                pre_proj=pre_proj,
                proj_layer_type=proj_layer_type,
                mix_noise=mix_noise,
            )
            simplenets.append(simplenet_inst)
        return simplenets

    return ("get_simplenet", get_simplenet)


# @main.command("dataset")
# @click.argument("name", type=str)
# @click.argument("data_path", type=click.Path(exists=True, file_okay=False))
# @click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
# @click.option("--train_val_split", type=float, default=1, show_default=True)
# @click.option("--batch_size", default=8, type=int, show_default=True)
# @click.option("--num_workers", default=2, type=int, show_default=True)
# @click.option("--resize", default=256, type=int, show_default=True)
# @click.option("--imagesize", default=224, type=int, show_default=True)
# @click.option("--rotate_degrees", default=0, type=int)
# @click.option("--translate", default=0, type=float)
# @click.option("--scale", default=0.0, type=float)
# @click.option("--brightness", default=0.0, type=float)
# @click.option("--contrast", default=0.0, type=float)
# @click.option("--saturation", default=0.0, type=float)
# @click.option("--gray", default=0.0, type=float)
# @click.option("--hflip", default=0.0, type=float)
# @click.option("--vflip", default=0.0, type=float)
# @click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    rotate_degrees,
    translate,
    scale,
    brightness,
    contrast,
    saturation,
    gray,
    hflip,
    vflip,
    augment,
):
    print("!"*100)
    print(f'name:{name}\n\
            data_path:{data_path}\n\
            subdatasets:{subdatasets}\n\
            train_val_split:{train_val_split}\n\
            batch_size:{batch_size}\n\
            resize:{resize}\n\
            imagesize:{imagesize}\n\
            num_workers:{num_workers}\n\
            rotate_degrees:{rotate_degrees}\n\
            translate:{translate}\n\
            scale:{scale}\n\
            brightness:{brightness}\n\
            contrast:{contrast}\n\
            saturation:{saturation}\n\
            gray:{gray}\n\
            hflip:{hflip}\n\
            vflip:{vflip}\n\
            augment:{augment}\n\
                        ')
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                rotate_degrees=rotate_degrees,
                translate=translate,
                brightness_factor=brightness,
                contrast_factor=contrast,
                saturation_factor=saturation,
                gray_p=gray,
                h_flip_p=hflip,
                v_flip_p=vflip,
                scale=scale,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )
            
            LOGGER.info(f"Dataset: train={len(train_dataset)} test={len(test_dataset)}")

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    prefetch_factor=4,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    # main()

    a1 = net(backbone_names=('wideresnet50',),
             layers_to_extract_from=('layer2', 'layer3'),
             pretrain_embed_dimension=1536,
             target_embed_dimension=1536,
             patchsize=3,
             embedding_size=256,
             meta_epochs=40,
             aed_meta_epochs=1,
             gan_epochs=4,
             noise_std=0.015,
             dsc_layers=2,
             dsc_hidden=1024,
             dsc_margin=0.5,
             dsc_lr=0.0002,
             auto_noise=0.0,
             train_backbone=False,
             cos_lr=False,
             pre_proj=1,
             proj_layer_type=0,
             mix_noise=1
        )
    a2 = dataset(name='mvtec',
                 data_path='d:/work/files/deeplearn_datasets/anomalydetection/test1',
                 subdatasets=('screw', 'pill', 'capsule', 'carpet', 'grid', 'tile', 'wood', 'zipper', 'cable', 'toothbrush', 'transistor', 'metal_nut', 'bottle', 'hazelnut', 'leather'),
                 train_val_split=1.0,
                 batch_size=8,
                 resize=329,
                 imagesize=288,
                 num_workers=2,
                 rotate_degrees=0,
                 translate=0,
                 scale=0.0,
                 brightness=0.0,
                 contrast=0.0,
                 saturation=0.0,
                 gray=0.0,
                 hflip=0.0,
                 vflip=0.0,
                 augment=False)
    methods=[a1,a2]
    
    run(methods=methods,
        results_path="results", 
        gpu=(1,),
        seed= 0, 
        log_group='simplenet_mvtec', 
        log_project='MVTecAD_Results', 
        run_name='run',
        test=False)
