class DefaultConfig(object):

    model = 'CNN'

    clip_grad = 2

    batch_size = 64
    num_epochs = 10

    num_class = 10
    lr = 0.001

    use_gpu = 1
    gpu_id = 0

    filters_num = 64

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)
        print('*************************************************')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print("{} => {}".format(k, getattr(self, k)))

        print('*************************************************')


opt = DefaultConfig()
