import torch
import torch.nn as nn 

def normalize_gradient(model, x):
    """
                     f
    f_hat = --------------------
            || grad_f || + | f |
    """
    x.requires_grad_(True)
    f = model(x)
    jac = []
    for i in range(f.shape[1]):
      grad, = torch.autograd.grad(
        f[:, i], [x], torch.ones_like(f[:, i]), create_graph=True, retain_graph=True)
      jac.append(grad)
    #jac [batch, classes, 3, h, w]
    jac = torch.stack(jac, 1)
  
    jac_norm = torch.norm(jac.view(jac.shape[0], jac.shape[1], -1), 
                    p=2, dim=-1)
   
    f_hat = f / (jac_norm + torch.abs(f))
    #f_hat = f / (jac_norm + 1e-12)
    return f_hat

class HingeLoss(nn.Module):
  def forward(self, pred_real, pred_fake=None):
    loss_real = F.relu(1 - pred_real).mean()
    loss_fake = F.relu(1 + pred_fake).mean()
    loss = loss_real + loss_fake
    return loss
   
