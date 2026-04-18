# blair/cf/run.py

import torch
import numpy as np

# Monkey-patch torch.randint so negative-sampling indices are always drawn on CPU
_real_randint = torch.randint

def _cpu_randint(*args, **kwargs):
    # Remove any device argument to prevent GPU indexing mismatches
    kwargs.pop('device', None)
    # Force device to CPU
    return _real_randint(*args, **{**kwargs, 'device': 'cpu'})

# Override global randint before importing RecBole
torch.randint = _cpu_randint

# Monkey-patch torch.load to set weights_only=False by default for compatibility
# with PyTorch 2.6+ and RecBole checkpoints
_real_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    # Set weights_only=False if not explicitly specified
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _real_torch_load(*args, **kwargs)

# Override global torch.load before importing RecBole
torch.load = _patched_torch_load

# RecBole still references deprecated NumPy aliases (e.g., np.float) removed in
# NumPy 2.0. Restore the small set they rely on so we can keep newer NumPy
# releases without pinning.
_NUMPY_BACKCOMP_ALIASES = {
    'float': float,
    'int': int,
    'bool': bool,
    'complex': complex,
    'object': object,
}
for _alias, _target in _NUMPY_BACKCOMP_ALIASES.items():
    # Use try-except to avoid the FutureWarning about np.object behavior change
    try:
        getattr(np, _alias)
    except AttributeError:
        setattr(np, _alias, _target)

from logging import getLogger
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color, get_trainer
# Direct import for the built-in RecBole UniSRec
from blair.cf.model.AlphaRec import AlphaRec #, AlphaRecTrainer
from blair.cf.data.dataset import AlphaRecDataset

def run_single(model_name, dataset, pretrained_file='', **kwargs):
    # configurations initialization
    props = ['blair/cf/config/overall.yaml',
            'blair/cf/config/AlphaRec.yaml']
    print(props)

    model_class = AlphaRec

    # configurations initialization
    config = Config(model=model_class, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = AlphaRecDataset(config)
    # dataset.build()
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device'])
    logger.info(model_name)

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }