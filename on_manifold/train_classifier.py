import sys
sys.path.append("..")
import datetime
import os
from os.path import join
import torch.utils.data as data
import argparse
import torch
from data import CelebA
from helpers import Progressbar
from on_manifold.model import Classifier
import torch.optim as optim
import torch.nn.functional as F


def parse(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--data', dest='data', type=str, choices=['CelebA', 'CelebA-HQ'], default='CelebA')
    parser.add_argument('--data_path', dest='data_path', type=str, default='../data/img_align_celeba')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='../data/list_attr_celeba.txt')
    parser.add_argument('--image_list_path', dest='image_list_path', type=str, default='data/image_list.txt')

    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)

    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--n_d', dest='n_d', type=int, default=5, help='# of d updates per g update')

    parser.add_argument('--b_distribution', dest='b_distribution', default='none',
                        choices=['none', 'uniform', 'truncated_normal'])
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--n_samples', dest='n_samples', type=int, default=16, help='# of sample images')

    parser.add_argument('--save_interval', dest='save_interval', type=int, default=5)
    parser.add_argument('--sample_interval', dest='sample_interval', type=int, default=5)
    parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true')
    parser.add_argument('--experiment_name', dest='experiment_name', default='classifier_eyeglass')
    return parser.parse_args(args)


use_gpu = torch.cuda.is_available()
attrs_default = ['Eyeglasses']
args = parse()
print(args)

args.lr_base = args.lr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

os.makedirs(join('output_classifier', args.experiment_name), exist_ok=True)
os.makedirs(join('output_classifier', args.experiment_name, 'checkpoint'), exist_ok=True)

train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'train', args.attrs)
valid_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'valid', args.attrs)
train_dataloader = data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    shuffle=True, drop_last=True
)
valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=args.n_samples, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)
print('Training images:', len(train_dataset), '/', 'Validating images:', len(valid_dataset))

classifier = Classifier(n_attrs=len(attrs_default))
if use_gpu: classifier.cuda()
optim_c = optim.Adam(classifier.parameters(), lr=args.lr, betas=args.betas)
progressbar = Progressbar()

fixed_img_a, fixed_att_a = next(iter(valid_dataloader))
fixed_img_a = fixed_img_a.cuda() if use_gpu else fixed_img_a
fixed_att_a = fixed_att_a.cuda() if use_gpu else fixed_att_a
fixed_att_a = fixed_att_a.type(torch.float)
sample_att_b_list = [fixed_att_a]

for epoch in range(args.epochs):
    # train with base lr in the first 100 epochs
    # and half the lr in the last 100 epochs
    total = correct = 0
    lr = args.lr_base / (10 ** (epoch // 100))
    for img_a, att_a in progressbar(train_dataloader):
        classifier.train()
        img_a = img_a.cuda() if use_gpu else img_a
        att_a = att_a.cuda() if use_gpu else att_a
        att_a = att_a.type(torch.float)
        output = classifier(img_a)
        prediction = (output >= 0.5)
        correct += (prediction == att_a).sum().item()
        total += len(att_a) * len(attrs_default)
        acc = correct / total
        c_loss = F.binary_cross_entropy_with_logits(output, att_a)
        classifier.zero_grad()
        c_loss.backward()
        optim_c.step()
        progressbar.say(epoch=epoch, c_loss=c_loss.item(), acc=acc)
    if epoch % args.save_interval == 0:
        torch.save(classifier.state_dict(), os.path.join(
            'output_classifier', args.experiment_name, 'checkpoint', 'weights.{:d}.pth'.format(epoch)
        ))
