from hamp_pred.src.models.common.models import BaseLinearWrapper


def get_seqs_vals(data, operator):
    if 'train_seq' not in data.columns:
        data['train_seq'] = data.apply(lambda x: x['n_seq'] + x['c_seq'], axis=1)
    params = ["rot", "shift_diff", "radius_diff", "pitch_diff"]
    seqs = list(data.train_seq.values)
    results = []
    for ind, row in data.iterrows():
        r = {param: list((row[param][0::2] + row[param][1::2]) / 2) for param in params}
        res = [list(zip(*list(r.values()))), list(zip(*list(r.values())))]
        results.append(res)
    train, valid, test = operator.get_for_train(seqs, results, test_size=0, valid_size=0)
    return train


def run(data, config=None):
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = config.get('n_chains', 2), 4
    operator.n_chains = n_chains
    valid_d = data[data['class'] == 'val']
    train_d = data[data['class'] != 'val']
    train, valid = get_seqs_vals(train_d, operator), get_seqs_vals(valid_d, operator)
    X, y = train
    val_x, val_y = valid
    y = y[:, :, 0:8]
    val_y = val_y[:, :, 0:8]
    model = model or BaseLinearWrapper(config=config)
    inp_shape = X.shape[1], X.shape[-1]
    model.build(inp_shape, 8).compile(optimizer="Adam", loss="mse", metrics=["mae"]). \
        train(X, y, val_x, val_y)
    return model
