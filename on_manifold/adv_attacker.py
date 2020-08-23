import torch
import torch.nn.functional as F


class FGSM():
    def __init__(self, decoder, classifier, eps=0.007):
        self.eps = eps
        self.decoder = decoder
        self.classifier = classifier

    def __call__(self, z, att, labels):
        att.requires_grad = True
        images = self.decoder(z, att)
        outputs = self.classifier(images)
        cost = F.binary_cross_entropy_with_logits(outputs, labels)

        grad = torch.autograd.grad(cost, att, retain_graph=False, create_graph=False)[0]

        adv_att = att + self.eps * grad.sign()
        adv_att = torch.clamp(adv_att, min=0, max=1).detach()

        return adv_att