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

from attgan import AttGAN
from on_manifold.model import Classifier
from utils import find_model
from on_manifold.adv_attacker import FGSM


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
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default='128_shortcut1_inject1_none_withoutglasses')
    return parser.parse_args(args)


attrs_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Male',
                 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young', 'Eyeglasses']
pos_classifier_att = 12

args = parse()
args.lr_base = args.lr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)
print(args)

# cuda
args.gpu = use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# load data
train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'train', args.attrs)
train_dataloader = data.DataLoader(
    train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)
# load classifier
classifier = Classifier(n_attrs=len(attrs_default))
if use_cuda: classifier.cuda()
classifier.load_state_dict(torch.load(find_model(os.path.join('output_classifier', 'checkpoint')), map_location=device))
classifier.eval()
# load AttGAN
attgan = AttGAN(args)
attgan.load(find_model(join('output', args.experiment_name, 'checkpoint')))
attgan.eval()
# load generator
encoder, decoder = attgan.G.encoder, attgan.G.decoder
fgsm = FGSM(decoder, classifier)

progressbar = Progressbar()
total = correct = 0
for img_a, att_a in progressbar(train_dataloader):
    img_a = img_a.cuda() if use_cuda else img_a
    att_a = att_a.cuda() if use_cuda else att_a
    att_a = att_a.type(torch.float)
    label = att_a.index_select(1, torch.tensor([pos_classifier_att]))
    att_a = att_a.index_select(1, torch.arange(len(attrs_default)-1))

    z = encoder(img_a)
    adv_att = fgsm(z, att_a, label)
    outputs = classifier(decoder(z, adv_att))
    prediction = (outputs >= 0.5)
    correct += (prediction == label).sum().item()
    total += (len(label)*len(attrs_default))
    acc = correct/total
    progressbar.say(acc=acc)
