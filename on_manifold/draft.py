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


def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--experiment_name', dest='experiment_name',
                        default='128_shortcut1_inject1_none_withoutglasses')
    return parser.parse_args(args)


attrs_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male',
                 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
args = parse()
print(args)

# cuda
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# load classifier
classifier = Classifier(n_attrs=len(attrs_default))
if use_cuda: classifier.cuda()
classifier.load_state_dict(torch.load(find_model(os.path.join('output_classifier', 'checkpoint')), map_location=device))
# load AttGAN
attgan = AttGAN(args)
attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))
# load generator
encoder, decoder = attgan.G.encoder, attgan.G.decoder







progressbar = Progressbar()
total = correct = 0
for img_a, att_a in progressbar(valid_dataloader):
    classifier.eval()
    img_a = img_a.cuda() if args.gpu else img_a
    att_a = att_a.cuda() if args.gpu else att_a
    att_a = att_a.type(torch.float)
    output = classifier(img_a)
    prediction = (output >= 0.5)
    correct += (prediction == att_a).sum().item()
    total += (len(att_a)*len(attrs_default))
    acc = correct/total
    progressbar.say(acc=acc)
