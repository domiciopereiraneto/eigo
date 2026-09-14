"""Categorical GA, linkage-tree GOMEA and independent random sampling."""
import time
import heapq
import numpy as np


def linkage_family(population, model='linkage_tree', block_size=2):
    dimension = population.shape[1]
    groups = [[i] for i in range(dimension)]
    if model == 'univariate':
        return groups
    if model == 'full':
        return [list(range(dimension))]
    if model == 'block_marginal_product':
        if block_size < 1:
            raise ValueError('gomea_bmp_block_size must be positive.')
        return [list(range(i, min(i + block_size, dimension))) for i in range(0, dimension, block_size)]
    if model not in {'linkage_tree', 'static_linkage_tree'}:
        raise ValueError('Unknown gomea_linkage_model.')
    # Categorical mutual information; token ID magnitude has no meaning.
    similarity = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(i):
            _, x = np.unique(population[:, i], return_inverse=True)
            _, y = np.unique(population[:, j], return_inverse=True)
            counts = np.zeros((x.max() + 1, y.max() + 1))
            np.add.at(counts, (x, y), 1)
            joint = counts / len(population)
            independent = joint.sum(1)[:, None] * joint.sum(0)[None, :]
            mask = joint > 0
            similarity[i, j] = similarity[j, i] = np.sum(joint[mask] * np.log(joint[mask] / independent[mask]))
    # Cache average-linkage similarities instead of rescanning all coordinate
    # pairs for every merge. Stale heap entries are discarded lazily.
    clusters = {i: g.copy() for i, g in enumerate(groups)}
    pair_scores = {(i, j): float(similarity[i, j])
                   for i in range(dimension) for j in range(i)}
    heap = [(-value, -i, -j) for (i, j), value in pair_scores.items()]
    heapq.heapify(heap)
    next_id = dimension
    while len(clusters) > 1:
        _, a, b = heapq.heappop(heap)
        a, b = -a, -b
        if a not in clusters or b not in clusters:
            continue
        left, right = clusters.pop(a), clusters.pop(b)
        merged = left + right
        for other in clusters:
            sa = pair_scores[max(a, other), min(a, other)]
            sb = pair_scores[max(b, other), min(b, other)]
            value = (len(left) * sa + len(right) * sb) / len(merged)
            pair_scores[next_id, other] = value
            heapq.heappush(heap, (-value, -next_id, -other))
        groups.append(merged)
        clusters[next_id] = merged
        next_id += 1
    return groups


def optimize(space, method, parameters, seed, evaluate, record):
    """Minimize evaluate(vector, generation); record each evaluated generation."""
    if method not in {'ga', 'gomea', 'random_sampler'}:
        raise ValueError('Unsupported optimization method.')
    rng = np.random.default_rng(seed)
    size = int(parameters.get('gomea_pop_size' if method == 'gomea' else 'pop_size', 16))
    generations = int(parameters.get('gomea_num_generations' if method == 'gomea' else 'num_generations', 15))
    if size < 2 or generations < 1:
        raise ValueError('Population size must be >= 2 and generations >= 1.')
    init_rate = float(parameters.get('token_init_rate', 0.1))
    mutation_rate = float(parameters.get('ga_mutation_rate', 0.1))
    crossover_rate = float(parameters.get('ga_crossover_rate', 0.5))
    elite_count = int(parameters.get('ga_elite_count', 2))
    if not 0 <= mutation_rate <= 1 or not 0 <= crossover_rate <= 1 or not 0 <= init_rate <= 1:
        raise ValueError('Token rates must be between 0 and 1.')
    if method == 'ga' and not 1 <= elite_count < size:
        raise ValueError('ga_elite_count must be >= 1 and smaller than pop_size.')
    if parameters.get('ga_crossover_operator', 'uniform') not in {'uniform', 'one_point'}:
        raise ValueError('ga_crossover_operator must be uniform or one_point.')
    if parameters.get('ga_mutation_operator', 'replacement') not in {'replacement', 'none'}:
        raise ValueError('ga_mutation_operator must be replacement or none.')
    budget = parameters.get('gomea_max_evaluations') if method == 'gomea' else None
    if method == 'random_sampler':
        budget = int(parameters.get('num_images_to_generate', 240))
    if budget is not None and int(budget) < 1:
        raise ValueError('Evaluation budget must be positive.')
    budget = int(budget) if budget is not None else None
    start = time.monotonic()
    limit = parameters.get('time_limit_seconds')
    best, best_score, evaluations = space.initial.copy(), float('inf'), 0
    def available():
        return (budget is None or evaluations < budget) and (limit is None or time.monotonic() - start < float(limit))
    def score(candidate, generation):
        nonlocal best, best_score, evaluations
        value = evaluate(candidate, generation)
        evaluations += 1
        if value < best_score:
            best, best_score = candidate.copy(), value
        return value
    # Baseline is recorded separately and is not charged to the search budget.
    best_score = evaluate(best, 0)
    record(0, best, best_score, 0)
    population = np.array([space.sample(rng, rate=init_rate) for _ in range(size)])
    population[0] = best
    scores = np.full(size, np.inf)
    static_family = None
    generation = 0
    while available() and (method == 'random_sampler' or generation < generations):
        generation += 1
        previous_evaluations = evaluations
        if method == 'random_sampler':
            for _ in range(min(size, budget - evaluations)):
                if not available(): break
                score(space.sample(rng), generation)
        elif generation == 1 or method == 'ga':
            for i in range(size):
                if not available(): break
                scores[i] = score(population[i], generation)
        else:
            model = parameters.get('gomea_linkage_model', 'linkage_tree')
            family = static_family or linkage_family(population, model, int(parameters.get('gomea_bmp_block_size', 2)))
            if model == 'static_linkage_tree': static_family = family
            donors = population.copy()
            for i in range(size):
                changed = False
                for subset_index in rng.permutation(len(family)):
                    if not available(): break
                    subset = family[subset_index]
                    trial = population[i].copy()
                    trial[subset] = donors[rng.integers(size), subset]
                    if np.array_equal(trial, population[i]): continue
                    value = score(trial, generation)
                    if value <= scores[i]:
                        changed |= not np.array_equal(trial, population[i])
                        population[i], scores[i] = trial, value
                # Forced improvement using the elitist donor if mixing stalls.
                if not changed:
                    for subset in family:
                        if not available(): break
                        trial = population[i].copy(); trial[subset] = best[subset]
                        if np.array_equal(trial, population[i]): continue
                        value = score(trial, generation)
                        if value < scores[i]:
                            population[i], scores[i] = trial, value
                            break
        if evaluations == previous_evaluations:
            break
        record(generation, best, best_score, evaluations)
        if method == 'ga':
            order = np.argsort(scores)
            next_population = [x.copy() for x in population[order[:elite_count]]]
            def parent():
                contestants = rng.integers(size, size=2)
                return population[contestants[np.argmin(scores[contestants])]]
            while len(next_population) < size:
                a, b = parent(), parent()
                mask = rng.random(len(best)) < crossover_rate
                if parameters.get('ga_crossover_operator', 'uniform') == 'one_point':
                    mask = np.arange(len(best)) < rng.integers(len(best) + 1)
                child = np.where(mask, a, b)
                rate = 0 if parameters.get('ga_mutation_operator') == 'none' else mutation_rate
                next_population.append(space.sample(rng, child, rate))
            population = np.array(next_population, dtype=np.int64)
            population[0] = best
    return best
