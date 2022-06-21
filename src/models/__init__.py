from .StructRankNetPQRS import StructRankNetPQRS

model_dict = {'StructRankNetPQRS':StructRankNetPQRS}

def allowed_models():
    return model_dict.keys()

def define_model(mod, args):
    if mod not in allowed_models():
        raise KeyError("The requested model: {} is not implemented".format(mod))
    else:
        return model_dict[mod](args)