class ArcFaceLoss(nn.Module):
    def __init__(self, scale, margin):
        super(ArcFaceLoss, self).__init__()
        self.scale = torch.tensor(scale)
        self.margin = torch.tensor(margin)
        self.cos_m = torch.cos(self.margin)
        self.sin_m = torch.sin(self.margin)
        self.th = torch.cos(torch.pi - self.margin)
        self.mm = torch.sin(torch.pi - self.margin) * self.margin

    def forward(self, logits, labels):
        # cosine of the angles between the feature vectors and weight vectors
        cosine = logits
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # phi is the target logit after adding the margin
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # convert labels to one-hot encoding
        one_hot = torch.zeros(cosine.size(), device=logits.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # apply the margin to the target logit
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        # use cross entropy as the loss function
        loss = F.cross_entropy(output, labels)

        return loss