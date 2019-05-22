def create_model(opt):
    if opt.model == 'DRN':
        from .DRN import DRN
        model = DRN()
        model.initialize(opt)
        model.setup()
    else:
        raise NotImplementedError('model [{}] is not found'.format(opt.model))

    print('model [{}] was created'.format(model.name()))
    return model
