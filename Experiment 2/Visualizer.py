import visdom
import numpy as np
import time
#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Visualizer(object):

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        """
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        # self.plot('loss', 1.00)

        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        """
        self.vis.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        self.function 等价于self.vis.function
        自定义的plot,image,log,plot_many等除外
        """
        return getattr(self.vis, name)
