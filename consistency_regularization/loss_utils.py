import torch
import torch.nn.functional as F

def _jensen_shannon_div(logit1, logit2, T=1.):
    prob1 = F.softmax(logit1/T, dim=1)
    prob2 = F.softmax(logit2/T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1, reduction='batchmean')
    jsd += F.kl_div(logsoftmax, prob2, reduction='batchmean')
    return jsd * 0.5

def ALP(imgs, imgs_adv, logit1, logit2):
  alp = torch.abs(logit1 - logit2)
  norm = torch.norm(imgs_adv.view(imgs_adv.shape[0], -1) - imgs.view(imgs.shape[0], -1),
               dim=1, keepdim=True)
  alp = alp / norm
 
  alp = torch.mean(torch.clamp(alp - 1, min=0.) **2)
  return alp
 
    