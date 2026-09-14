"""Integer search spaces captured at the pipeline's text encoder boundary."""
from contextlib import contextmanager
import numpy as np
import torch


def resolve_optimization_target(parameters):
    target = parameters.get('optimization_target', 'text_tokens')
    if target != 'text_tokens':
        raise ValueError('Only optimization_target: text_tokens is supported.')
    return target


class TokenSpace:
    """Preserve backend tokenization, masks, special tokens and pooling behavior.

    Each encoder has its own vocabulary and mutable coordinates. Replacement
    occurs only on its first (positive prompt) forward call. The pipeline still
    handles negative prompts and all backend-specific embedding transformations.
    """
    def __init__(self, engine, prompt):
        self.engine, self.prompt = engine, prompt
        self.parts = []
        handles = []
        try:
            for suffix in ('', '_2', '_3'):
                encoder = getattr(engine.pipe, 'text_encoder' + suffix, None)
                tokenizer = getattr(engine.pipe, 'tokenizer' + suffix, None)
                if encoder is None or tokenizer is None:
                    continue
                part = dict(name='text_encoder' + suffix, encoder=encoder, tokenizer=tokenizer)
                self.parts.append(part)
                def capture(module, args, kwargs, part=part):
                    if 'ids' not in part:
                        ids = kwargs.get('input_ids', args[0] if args else None)
                        if ids is None or ids.ndim != 2 or ids.shape[0] != 1:
                            raise ValueError('Expected one tokenized positive prompt per encoder.')
                        part['ids'] = ids.detach().clone()
                        part['mask'] = kwargs.get('attention_mask')
                handles.append(encoder.register_forward_pre_hook(capture, with_kwargs=True))
            with torch.no_grad():
                engine._encode_prompt_embeddings(prompt)
        finally:
            for handle in handles:
                handle.remove()
        self.parts = [p for p in self.parts if 'ids' in p]
        if not self.parts:
            raise ValueError('No supported text encoder input_ids were captured.')
        initial, self.domains = [], []
        for part in self.parts:
            ids = part['ids'].cpu().numpy().reshape(-1)
            specials = part['tokenizer'].all_special_ids
            mutable = ~np.isin(ids, specials)
            if part['mask'] is not None:
                mutable &= part['mask'].detach().cpu().numpy().reshape(-1).astype(bool)
            part['positions'] = np.flatnonzero(mutable)
            vocab = part['tokenizer'].get_vocab().values()
            size = part['encoder'].get_input_embeddings().num_embeddings
            allowed = np.array(sorted(set(vocab) - set(specials)), dtype=np.int64)
            allowed = allowed[(allowed >= 0) & (allowed < size)]
            if not len(allowed):
                raise ValueError('Encoder vocabulary has no non-special tokens.')
            part['start'] = len(initial)
            initial.extend(ids[mutable].tolist())
            self.domains.extend([allowed] * int(mutable.sum()))
            part['end'] = len(initial)
        self.initial = np.asarray(initial, dtype=np.int64)
        if not len(self.initial):
            raise ValueError('Prompt has no mutable non-special tokens.')

    def validate(self, vector):
        vector = np.asarray(vector)
        if vector.shape != self.initial.shape or not np.issubdtype(vector.dtype, np.integer):
            raise ValueError('Candidate must be an integer vector matching the token space.')
        for value, domain in zip(vector, self.domains):
            i = np.searchsorted(domain, value)
            if i == len(domain) or domain[i] != value:
                raise ValueError(f'Invalid token ID: {value}')
        return vector

    def sample(self, rng, base=None, rate=1.0):
        if not 0 <= rate <= 1:
            raise ValueError('Token replacement rate must be between 0 and 1.')
        vector = self.initial.copy() if base is None else self.validate(base).copy()
        for i in np.flatnonzero(rng.random(len(vector)) < rate):
            vector[i] = rng.choice(self.domains[i])
        return vector

    def token_ids(self, vector, part):
        ids = part['ids'].clone()
        ids[0, part['positions']] = torch.as_tensor(
            vector[part['start']:part['end']], device=ids.device, dtype=ids.dtype)
        return ids

    @contextmanager
    def inject(self, vector):
        vector = self.validate(vector)
        handles, seen = [], set()
        try:
            for part in self.parts:
                def replace(module, args, kwargs, part=part):
                    if part['name'] in seen:
                        return
                    seen.add(part['name'])
                    ids = self.token_ids(vector, part)
                    if 'input_ids' in kwargs:
                        kwargs = dict(kwargs, input_ids=ids)
                    else:
                        args = (ids,) + args[1:]
                    return args, kwargs
                handles.append(part['encoder'].register_forward_pre_hook(replace, with_kwargs=True))
            yield
            if len(seen) != len(self.parts):
                raise RuntimeError('Pipeline did not call every captured text encoder.')
        finally:
            for handle in handles:
                handle.remove()

    def encode(self, vector):
        with torch.no_grad(), self.inject(vector):
            return self.engine._encode_prompt_embeddings(self.prompt)

    def artifact(self, vector):
        vector = self.validate(vector)
        return {'optimization_target': 'text_tokens', 'original_prompt': self.prompt,
                'tokens': vector.tolist(), 'encoders': {
                    p['name']: {'input_ids': self.token_ids(vector, p)[0].cpu().tolist(),
                                'mutable_positions': p['positions'].tolist(),
                                'decoded_text': p['tokenizer'].decode(self.token_ids(vector, p)[0].cpu().tolist())}
                    for p in self.parts}}
