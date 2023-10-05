from hamp_pred.src.models.common.models import BaseLinearWrapper


def get_seqs_vals(data, operator):
    if 'train_seq' not in data.columns:
        data['train_seq'] = data.apply(lambda x: x['n_seq'] + x['c_seq'], axis=1)
    params = ["crick_mut", "shift", "radius", "pitch", "P"]
    seqs = list(data.train_seq.values)
    results = []
    for ind, row in data.iterrows():
        res = []
        for helix in ['n', 'c']:
            r = {param: list((row[helix + "_" + param][0::2] + row[helix + "_" +param][1::2]) / 2) for param in params}
            res.append(list(zip(*list(r.values()))))
        results.append(res)
    train, valid, test = operator.get_for_train(seqs, results, test_size=0, valid_size=0)
    return train


def run(data, config=None):
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = config.get('n_chains', 2), 10
    operator.n_chains = n_chains
    valid_d = data[data['class'] == 'val']
    train_d = data[data['class'] != 'val']
    train, valid = get_seqs_vals(train_d, operator), get_seqs_vals(valid_d, operator)
    X, y = train
    val_x, val_y = valid
    y = y[:, :, 0:20]
    val_y = val_y[:, :, 0:20]
    model = model or BaseLinearWrapper(config=config)
    inp_shape = X.shape[1], X.shape[-1]
    model.build(inp_shape, 20).compile(optimizer="Adam", loss="mse", metrics=["mae"]). \
        train(X, y, val_x, val_y)
    return model
