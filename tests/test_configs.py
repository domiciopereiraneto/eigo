import contextlib
import io
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'algorithms'))
from eigo_grid_search import build_run_queue, create_run_config, extract_final_results
from eigo_optuna_search import validate_optuna_config, get_enabled_methods, objective_metric
from process_runchart_results import clean_curve_arrays


class ConfigTests(unittest.TestCase):
    def test_all_yaml_and_search_methods(self):
        for path in Path('algorithms/config').glob('*.yaml'):
            config=yaml.safe_load(path.read_text())
            self.assertIsInstance(config,dict)
            if 'model_id' in config: self.assertEqual(config['optimization_target'],'text_tokens')
        grid=yaml.safe_load(Path('algorithms/config/config_eigo_grid_search.yaml').read_text())
        self.assertEqual({m for m,_ in build_run_queue(grid)},{'ga','gomea','random_sampler'})
        config=yaml.safe_load(Path('algorithms/config/config_eigo_optuna_search.yaml').read_text())
        validate_optuna_config(config)
        self.assertEqual(set(get_enabled_methods(config)),{'ga','gomea','random_sampler'})
        self.assertEqual(objective_metric({'status':'ok','best_fitness':8,'max_fitness':3},'random_sampler','auto'),(8.,'best_fitness'))
    def test_gomea_actual_evaluation_accounting(self):
        frame=pd.DataFrame({'generation':[0,1,2],'evaluations':[0,4,17],
                            'max_fitness':[1,2,3],'elapsed_time':[0,1,2]})
        _, nfe, _, _=clean_curve_arrays(frame,'generation','max_fitness','objective',True,'fitness',4,10)
        np.testing.assert_array_equal(nfe,[10,50,180])

if __name__=='__main__': unittest.main()
