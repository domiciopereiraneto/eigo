import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.optimization_targets import TokenSpace, resolve_optimization_target
from src.token_optimizers import optimize, linkage_family


class Tokenizer:
    all_special_ids = [0, 1, 2]
    def __init__(self, size): self.size = size
    def get_vocab(self): return {str(i): i for i in range(self.size)}
    def decode(self, ids): return ' '.join(map(str, ids))


class Encoder(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.embedding = torch.nn.Embedding(size, 2)
        self.seen = []
    def get_input_embeddings(self): return self.embedding
    def forward(self, input_ids, attention_mask=None):
        self.seen.append(input_ids.clone())
        return self.embedding(input_ids)


class Engine:
    def __init__(self):
        self.pipe = SimpleNamespace(text_encoder=Encoder(8), tokenizer=Tokenizer(8),
                                    text_encoder_2=Encoder(12), tokenizer_2=Tokenizer(12))
    def _encode_prompt_embeddings(self, prompt):
        outputs = []
        for suffix in ('', '_2'):
            encoder = getattr(self.pipe, 'text_encoder' + suffix)
            outputs.append(encoder(input_ids=torch.tensor([[1, 3, 4, 2, 0]]), attention_mask=torch.tensor([[1, 1, 1, 1, 0]])))
            encoder(torch.tensor([[1, 2, 0, 0, 0]]))
        return outputs[0], outputs[1]


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine()
        self.space = TokenSpace(self.engine, 'prompt')
    def test_exact_ids_and_negative_special_padding_preserved(self):
        self.space.encode(np.array([7, 6, 11, 10]))
        self.assertEqual(self.engine.pipe.text_encoder.seen[-2].tolist(), [[1, 7, 6, 2, 0]])
        self.assertEqual(self.engine.pipe.text_encoder_2.seen[-2].tolist(), [[1, 11, 10, 2, 0]])
        self.assertEqual(self.engine.pipe.text_encoder.seen[-1].tolist(), [[1, 2, 0, 0, 0]])
        self.assertFalse(self.engine.pipe.text_encoder._forward_pre_hooks)
    def test_invalid_ids_and_floats_rejected(self):
        for vector in [np.array([8, 3, 4, 5]), np.array([1, 3, 4, 5]), np.ones(4), np.ones(2, dtype=int)]:
            with self.assertRaises(ValueError): self.space.validate(vector)
        with self.assertRaises(ValueError): resolve_optimization_target({'optimization_target':'latent_noise'})
    def test_hooks_removed_on_exception(self):
        with self.assertRaises(RuntimeError):
            with self.space.inject(self.space.initial): raise RuntimeError('test')
        self.assertFalse(self.engine.pipe.text_encoder._forward_pre_hooks)
    def test_reproducible_search_and_budget(self):
        for method in ['ga', 'gomea', 'random_sampler']:
            def run():
                trace, records = [], []
                def evaluate(x, generation):
                    self.space.validate(x)
                    trace.append((generation, x.tolist()))
                    return float(-x.sum())
                result = optimize(self.space, method, {'pop_size':6,'gomea_pop_size':6,
                    'num_generations':4,'gomea_num_generations':4,'gomea_max_evaluations':11,
                    'num_images_to_generate':11}, 42, evaluate,
                    lambda g, x, f, n: records.append((g, f, n)))
                return result, trace, records
            a, trace, records = run(); b, trace2, _ = run()
            np.testing.assert_array_equal(a,b); self.assertEqual(trace,trace2)
            self.assertTrue(all(records[i][1] <= records[i-1][1] for i in range(1,len(records))))
            if method != 'ga': self.assertLessEqual(len(trace)-1,11)
            if method == 'random_sampler': self.assertEqual(len(trace)-1,11)
    def test_linkage_is_categorical(self):
        pop=np.array([[3,3,4],[3,3,5],[4,4,4],[4,4,5]])
        family=linkage_family(pop)
        self.assertIn([1,0],family)
        self.assertEqual(sorted(family[-1]),[0,1,2])
    def test_deadline_keeps_baseline(self):
        trace=[]
        result=optimize(self.space,'ga',{'time_limit_seconds':0},1,
                        lambda x,g: trace.append(g) or 0,lambda *args:None)
        self.assertEqual(trace,[0]); np.testing.assert_array_equal(result,self.space.initial)
    def test_end_to_end_artifacts_without_models(self):
        from eigo import Eigo
        class FakeEigo(Eigo):
            def _encode_prompt_embeddings(inner,prompt):
                return self.engine._encode_prompt_embeddings(prompt)
            def generate_image_from_tensors(inner, pe, pooled, seed, latents=None):
                return torch.full((8,8,3), float(torch.sigmoid(pe.mean())))
            def _evaluate_canonical_image_scores(inner,image,prompt,jpeg_size_kb=None):
                value=float(image.mean())
                return value, value, value, value, value, value, jpeg_size_kb or 1., {}
            def _save_population_plot_results(inner,*args): pass
            def _save_run_config(inner,folder,seed,prompt,*args):
                Path(folder,'config.yaml').write_text('optimization_target: text_tokens\n')
        for method in ['ga','gomea','random_sampler']:
            with tempfile.TemporaryDirectory() as folder:
                engine=FakeEigo.__new__(FakeEigo)
                engine.pipe=self.engine.pipe;engine.device='cpu';engine.jpeg_quality=95
                engine.OUTPUT_FOLDER=folder;engine.model_name='test'
                engine.parameters={'seed':42,'selected_prompt':'prompt','pop_size':4,'num_generations':2,
                    'gomea_pop_size':4,'gomea_num_generations':2,'num_images_to_generate':5,'save_gens':True}
                output=Path(engine._run_token_optimization(method))
                artifact=json.loads((output/'best_tokens.json').read_text())
                self.space.validate(np.array(artifact['tokens']))
                self.assertTrue((output/'best_all.jpg').exists())
                self.assertTrue((output/'fitness_results.csv').exists())
                self.assertTrue((output/'candidates.jsonl').exists())

if __name__=='__main__': unittest.main()
