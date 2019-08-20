from tv_abc import TVBase


class Trainer(TVBase):
    def __init__(self, mode, meters, device):
        super().__init__(mode, meters, device)

    def set_net_mode(self, net):
        net.train()

    def batch_main(self, net, x_slow, x_fast, annotation):
        p = net.forward((x_slow, x_fast))

        a = self.generate_gt(annotation)

        loss = net.loss_update(p, a, step=True)

        acc = self.evaluate(p, a)

        return {'loss': loss,
                'label_accuracy': acc}
