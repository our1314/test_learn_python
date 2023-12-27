#https://zhuanlan.zhihu.com/p/112394326

import click

@click.group(chain=True)

@click.command('m1')
@click.option("-n1", "--num1", help="input a num")
@click.option("-k1", "--kk1", help="input a kk")
def m1(num1, kk1):
    print(f'input={num1}')
    print(f'input={kk1}')


@click.command('m2')
@click.option("-n2", "--num2", help="input a num")
@click.option("-k2", "--kk2", help="input a kk")
def m2(num2, kk2):
    print(f'input={num2}')
    print(f'input={kk2}')

def main(**kwargs):
    
    pass

if __name__ == "__main__":
    main()
    pass


backbone_names:('wideresnet50',)
layers_to_extract_from:('layer2', 'layer3')
pretrain_embed_dimension:1536
target_embed_dimension:1536
patchsize:3
embedding_size:256
meta_epochs:200
aed_meta_epochs:1
gan_epochs:4
noise_std:0.015
dsc_layers:2
dsc_hidden:1024
dsc_margin:0.5
dsc_lr:0.0002
auto_noise:0.0
train_backbone:False
cos_lr:False
pre_proj:1
proj_layer_type:0
mix_noise:1