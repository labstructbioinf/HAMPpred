from hamp_pred.src.models.common.models import BaseLinearWrapper


def run(data, config=None):
    model, operator = config.get('model')(config=config.get('model_config')), config.get('operator')
    n_chains, features = 1, 2
    operator.n_chains = n_chains
    seqs = list(data.full_sequence.values)
    vals = [[[[c] for c in list(x)]] for x in data['full_crick'].values]
    train, valid, test = operator.get_for_train(seqs, vals, ids=config.get('ids'),
                                                test_ids=config.get('test_ids'),
                                                val_ids=config.get('val_ids'))
    X, y = train
    val_x, val_y = valid
    model = model or BaseLinearWrapper(config=config)
    inp_shape = X.shape[1], X.shape[-1]
    model.build(inp_shape, features).compile(optimizer="Adam", loss="mse", metrics=["mae"]). \
        train(X, y, val_x, val_y)
    return model
