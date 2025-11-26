import os
import sys
import time
import json
import heapq
import argparse

import numpy as np
import pandas as pd
from ViennaRNA import RNA

from utils.vienna import position_ed_pd_mfe, position_ed_ned_mfe, mfe, delta_delta_energy
from utils.structure import extract_pairs, struct_dist
from utils.constants import P1, P2, U1, U2, IUPAC_CODES

# from multiprocessing import Pool, cpu_count


name2pair = {'cg': ['CG', 'GC'],
             'cggu': ['CG', 'GC', 'GU', 'UG'],
             'cgau': ['CG', 'GC', 'AU', 'UA'],
             'all': ['CG', 'GC', 'AU', 'UA', 'GU', 'UG']}

nuc_others = {'A': 'CGU',
              'C': 'AGU',
              'U': 'ACG',
              'G': 'ACU'}

nuc_pair_others = {'AU': ['UA', 'CG', 'GC', 'UG', 'GU'],
                   'UA': ['AU', 'CG', 'GC', 'UG', 'GU'],
                   'CG': ['AU', 'UA', 'GC', 'UG', 'GU'],
                   'GC': ['AU', 'UA', 'CG', 'UG', 'GU'],
                   'GU': ['AU', 'UA', 'CG', 'GC', 'UG'],
                   'UG': ['AU', 'UA', 'CG', 'GC', 'GU']}

nuc_all = ['A', 'C', 'G', 'U']
nuc_pair_all = ['AU', 'UA', 'CG', 'GC', 'UG', 'GU']

STAY = 2000
STOP = 0.001
MAX_REPEAT = 1000
FREQ_PRINT = 10

WORKER_COUNT = 10
BATCH_SIZE = 20
pair_cache = {}
LOG = False


class RNAStructure:

    def __init__(self, seq, score, v=None,
                 v_list=None):  # v_list: positional NED, v: objective value, socore: used for priority queue
        self.seq = seq
        self.score = score
        self.v = v
        self.v_list = v_list

    def __gt__(self, other):
        return self.score > other.score

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.seq == other.seq

    def __ge__(self, other):
        return self.score >= other.score

    def __le__(self, other):
        return self.score <= other.score

    def __str__(self):
        return f"{self.seq}: {self.score: .4f}, {1 - self.score: .4f}"

    def __repr__(self):
        return f"RNAStructure('{self.seq}', {self.score})"

    def __hash__(self):
        return hash(self.seq)


import heapq  # Make sure to import heapq at the top of your file


def calculate_gc_bounds(target_structure, constraints, pairs_map):
    """
    Calculates the minimum and maximum possible GC content for a sequence
    given a target structure and IUPAC constraints.

    Args:
        target_structure (str): The target structure in dot-bracket notation.
        constraints (str): The sequence constraints in IUPAC notation.
        pairs_map (dict): A pre-computed dictionary mapping paired indices.

    Returns:
        tuple: A tuple containing (minimum_gc_count, maximum_gc_count).
    """
    min_gc_count = 0
    max_gc_count = 0
    processed_indices = set()

    for i, structural_char in enumerate(target_structure):
        if i in processed_indices:
            continue

        # --- Handle Unpaired Positions ---
        if structural_char == '.':
            allowed_nucs = IUPAC_CODES[constraints[i]]

            # For MINIMUM, check if we can AVOID G/C
            non_gc_options = [n for n in allowed_nucs if n not in ['G', 'C']]
            if not non_gc_options:
                min_gc_count += 1  # Must use G or C

            # For MAXIMUM, check if we can CHOOSE G/C
            gc_options = [n for n in allowed_nucs if n in ['G', 'C']]
            if gc_options:
                max_gc_count += 1  # Can use G or C

            processed_indices.add(i)

        # --- Handle Paired Positions ---
        elif structural_char == '(':
            j = pairs_map[i]

            # Find all possible pairs that satisfy the IUPAC constraints
            key = (constraints[i], constraints[j])
            if key in pair_cache:
                possible_pairs = pair_cache[key]
            else:  # Should be cached, but calculate as a fallback
                allowed_i = IUPAC_CODES[constraints[i]]
                allowed_j = IUPAC_CODES[constraints[j]]
                possible_pairs = [p for p in nuc_pair_all if p[0] in allowed_i and p[1] in allowed_j]
                pair_cache[key] = possible_pairs

            if not possible_pairs:
                raise ValueError("Impossible constraints for pair.")

            # For MINIMUM, check if we can AVOID GC/CG pairs
            non_gc_pairs = [p for p in possible_pairs if p not in ['CG', 'GC']]
            if non_gc_pairs:
                # Check if we can choose AU/UA which have 0 GC
                if any(p in ['AU', 'UA'] for p in non_gc_pairs):
                    pass  # min_gc_count increases by 0
                else:  # Must choose GU/UG
                    min_gc_count += 1
            else:  # Must choose CG/GC
                min_gc_count += 2

            # For MAXIMUM, check if we can CHOOSE GC/CG pairs
            gc_pairs = [p for p in possible_pairs if p in ['CG', 'GC']]
            if gc_pairs:
                max_gc_count += 2  # Can choose CG or GC
            elif any(p in ['GU', 'UG'] for p in possible_pairs):
                max_gc_count += 1  # Can choose GU or UG
            else:
                pass  # Must choose AU or UA, max_gc_count increases by 0

            processed_indices.add(i)
            processed_indices.add(j)

    return min_gc_count/len(target)*100, max_gc_count/len(target)*100
def analyze_and_format_k_best(k_best_heap, target_structure):
    """
    Analyzes each sequence in the k_best heap and formats the results for CSV output.
    """
    analysis_data = []

    # --- Correct and Efficient Sorting Logic for a Heap ---
    # 1. Pop all items from the min-heap. This gives a list sorted by score (worst to best).
    sorted_list = []
    while k_best_heap:
        sorted_list.append(heapq.heappop(k_best_heap))

    # 2. Reverse the list to get the final order from best (highest score) to worst.
    sorted_list.reverse()

    print("\n--- Analyzing Top K Results ---")
    for i, rna_struct in enumerate(sorted_list):
        seq = rna_struct.seq

        # --- The rest of the detailed analysis logic remains the same ---
        fc = RNA.fold_compound(seq)

        # --- 1. Calculate Partition Function and MFE ---
        (pf_struct, pf_energy) = fc.pf()
        (mfe_struct, mfe_energy) = fc.mfe()

        # --- 2. Calculate Equilibrium Probability ---
        probability = fc.pr_structure(target_structure)
        prob_defect = probability
        ned = fc.ensemble_defect(target_structure)
        ed = ned * len(seq)
        ddg = delta_delta_energy(seq, target_structure)
        dist = struct_dist(target_structure, mfe_struct)
        is_mfe = (dist == 0)

        # Check for uMFE status
        mfe_count = 0
        for s in RNA.subopt(seq, 0):
            if abs(s.energy - mfe_energy) < 1e-6:
                mfe_count += 1
            else:
                break
        is_umfe = is_mfe and (mfe_count == 1)

        # Append the formatted data
        analysis_data.append({
            "top i": i + 1,
            "seq": seq,
            "pd": prob_defect,
            "NED": ned,
            "ED": ed,
            "delta delta G": ddg,
            "struct distance": dist,
            "is MFE": is_mfe,
            "is uMFE": is_umfe
        })
    print("Analysis complete.")
    return analysis_data
def init_with_pair(t, pos_pairs, pairs_init):
    rna = list("." * len(t))
    assert len(rna) == len(t)
    for i, s in enumerate(t):
        if s == ".":
            rna[i] = 'A'
            if name_pair == 'all':
                rna[i] = np.random.choice(['A', 'C', 'G', 'U'])
        elif s == "(":
            j = pos_pairs[i]
            pair = np.random.choice(pairs_init)
            rna[i] = pair[0]
            rna[j] = pair[1]
        elif s == ")":
            pass
        else:
            raise ValueError(f'the value of structure at position: {i} is not right: {s}!')
    return "".join(rna)


# Added for partial RNA design using IUPAC constraints
def init_with_pair_constrained(t, pos_pairs, constraints):
    """
    Generates an initial RNA sequence that conforms to a target structure 't'
    and a set of IUPAC 'constraints'. Uses a global cache for pair calculations
    and prioritizes 'CG'/'GC' pairs and 'A' for unpaired positions.
    """
    global pair_cache  # Declare that we will be using the global cache
    rna = list("." * len(t))
    assert len(rna) == len(t)

    for i, s in enumerate(t):
        if rna[i] != ".":
            continue

        if s == ".":
            allowed_nucs = IUPAC_CODES[constraints[i]]
            if 'A' in allowed_nucs:
                rna[i] = 'A'
            else:
                rna[i] = np.random.choice(allowed_nucs)

        elif s == "(":
            j = pos_pairs[i]

            # --- Caching Logic ---
            code_i = constraints[i]
            code_j = constraints[j]
            cache_key = (code_i, code_j)

            if cache_key in pair_cache:
                possible_pairs = pair_cache[cache_key]
            else:
                allowed_i = IUPAC_CODES[code_i]
                allowed_j = IUPAC_CODES[code_j]
                calculated_pairs = [p for p in nuc_pair_all if p[0] in allowed_i and p[1] in allowed_j]
                pair_cache[cache_key] = calculated_pairs
                possible_pairs = calculated_pairs

            if not possible_pairs:
                raise ValueError(
                    f"Impossible constraints at pair ({i}, {j}): structure requires a pair, but IUPAC codes '{constraints[i]}' and '{constraints[j]}' allow no valid pairings.")

            # --- Prioritization Logic ---
            high_priority_pairs1 = [p for p in possible_pairs if p in ['CG', 'GC']]
            high_priority_pairs2 = [p for p in possible_pairs if p in ['AU', 'UA']]

            if high_priority_pairs1:
                pair = np.random.choice(high_priority_pairs1)
            elif high_priority_pairs2:
                pair = np.random.choice(high_priority_pairs2)
            else:
                pair = np.random.choice(possible_pairs)

            rna[i] = pair[0]
            rna[j] = pair[1]

        elif s != ")":
            raise ValueError(f'The value of structure at position: {i} is not right: {s}!')

    return "".join(rna)


# targeted initilization
def init_k(target, pos_pairs, k):
    print(f'name_pair: {name_pair}')
    pair_pool = name2pair[name_pair]
    print(f'pair_pool: {pair_pool}')
    init_0 = init_with_pair(target, pos_pairs, pair_pool)
    p_list = [init_0]
    # if too few pairs then use 'cggu', however this may never happen
    if k > len(pair_pool) ** (len(pos_pairs) / 2) and len(pair_pool) < 4:
        pair_pool = name2pair['cggu']
    # the max number of intial sequences is: len(pair_pool)**(len(pos_pairs)/2)
    while len(p_list) < min(k, len(pair_pool) ** (len(pos_pairs) / 2)):
        init_i = init_with_pair(target, pos_pairs, pair_pool)
        if init_i not in p_list:
            p_list.append(init_i)
    return p_list


# Added for partial RNA design using IUPAC constraints
def init_k_constrained(target, pos_pairs, k, constraints):
    # Generate the first sequence using our new constrained function
    init_0 = init_with_pair_constrained(target, pos_pairs, constraints)
    p_list = [init_0]

    # A simple counter to prevent an infinite loop with very restrictive constraints
    attempts = 0
    max_attempts = k * 100  # A reasonable safety limit

    # Loop until we have k unique sequences or we time out
    while len(p_list) < k and attempts < max_attempts:
        init_i = init_with_pair_constrained(target, pos_pairs, constraints)
        if init_i not in p_list:
            p_list.append(init_i)
        attempts += 1

    if len(p_list) < k:
        print(
            f"Warning: Could only generate {len(p_list)} of {k} requested unique sequences due to the provided constraints.")

    return p_list


def pairs_match(ss):  # find the pairs in a secondary structure, return a dictionary
    assert len(ss) > 5
    pairs = dict()
    stack = []
    for i, s in enumerate(ss):
        if s == ".":
            pass
        elif s == "(":
            stack.append(i)
        elif s == ")":
            j = stack.pop()
            assert j < i
            pairs[j] = i
            pairs[i] = j
        else:
            raise ValueError(f'the value of structure at position: {i} is not right: {s}!')
    return pairs


def mutate_pair(nuc_i, nuc_j, exclude=False):
    pair_ij = nuc_i + nuc_j
    return np.random.choice(nuc_pair_others[pair_ij]) if exclude else np.random.choice(nuc_pair_all)


def mutate_unpair(nuc_i, exclude=False):
    return np.random.choice(list(nuc_others[nuc_i])) if exclude else np.random.choice(nuc_all)


# traditional mutation
def mutate_tradition(seq, pairs, v, v_list, T, pairs_dg=None):
    v_list = [v / T for v in v_list]
    probs = np.exp(v_list) / sum(np.exp(v_list))
    index = np.random.choice(list(range(len(seq))), p=probs)
    seq_next = [nuc for nuc in seq]
    if index in pairs:
        i = min(index, pairs[index])
        j = max(index, pairs[index])
        pair_ij = seq[i] + seq[j]
        pair_new = np.random.choice(nuc_pair_others[pair_ij])
        seq_next[i] = pair_new[0]
        seq_next[j] = pair_new[1]
    else:
        c = np.random.choice(list(nuc_others[seq[index]]))
        assert c != seq[index]
        seq_next[index] = c
    return "".join(seq_next)


# Added for partial RNA design using IUPAC constraints
def mutate_tradition_constrained(seq, pairs, v, v_list, T, constraints):
    v_list = [v / T for v in v_list]
    probs = np.exp(v_list) / sum(np.exp(v_list))
    index = np.random.choice(list(range(len(seq))), p=probs)
    seq_next = list(seq)

    if index in pairs:
        # --- Paired Mutation Logic ---
        i = min(index, pairs[index])
        j = max(index, pairs[index])

        # Find all possible valid pairs that satisfy the constraints using the global cache
        code_i = constraints[i]
        code_j = constraints[j]
        cache_key = (code_i, code_j)
        if cache_key in pair_cache:
            possible_pairs = pair_cache[cache_key]
        else:  # Should have been cached during init, but calculate just in case
            allowed_i = IUPAC_CODES[code_i]
            allowed_j = IUPAC_CODES[code_j]
            possible_pairs = [p for p in nuc_pair_all if p[0] in allowed_i and p[1] in allowed_j]
            pair_cache[cache_key] = possible_pairs

        current_pair = seq[i] + seq[j]
        choices = [p for p in possible_pairs if p != current_pair]

        if choices:  # If other options exist
            pair_new = np.random.choice(choices)
            seq_next[i] = pair_new[0]
            seq_next[j] = pair_new[1]

    else:
        # --- Unpaired Mutation Logic ---
        allowed_nucs = IUPAC_CODES[constraints[index]]
        choices = [n for n in allowed_nucs if n != seq[index]]

        if choices:  # If other options exist
            c = np.random.choice(choices)
            seq_next[index] = c

    return "".join(seq_next)


# structured mutation
def mutate_structured(seq, pairs, v, v_list, T):
    v_list = [v / T for v in v_list]
    probs = np.exp(v_list) / sum(np.exp(v_list))
    index = np.random.choice(list(range(len(seq))), p=probs)
    pairs_mt = []
    unpairs_mt = []

    if index in pairs:
        i = min(index, pairs[index])
        j = max(index, pairs[index])
        pairs_mt.append((i, j))
        if j - 1 in pairs and pairs[j - 1] == i + 1:
            pairs_mt.append((pairs[j - 1], j - 1))
            if i + 2 not in pairs and j - 2 not in pairs:
                unpairs_mt.append(i + 2)
                unpairs_mt.append(j - 2)
        if i + 1 not in pairs and j - 1 not in pairs:
            unpairs_mt.append(i + 1)
            unpairs_mt.append(j - 1)
    else:
        unpairs_mt.append(index)
        if index - 1 in pairs and pairs[index - 1] > index:
            pairs_mt.append((index - 1, pairs[index - 1]))
            if pairs[index - 1] - 1 not in pairs:
                unpairs_mt.append(pairs[index - 1] - 1)
        elif index + 1 in pairs and pairs[index + 1] < index:
            pairs_mt.append((pairs[index + 1], index + 1))
            if pairs[index + 1] + 1 not in pairs:
                unpairs_mt.append(pairs[index + 1] + 1)

    assert len(pairs_mt) <= 2, pairs_mt
    assert len(unpairs_mt) <= 2, unpairs_mt

    # one pair
    if len(pairs_mt) == 1:
        pairs_selected_index = np.random.choice(range(len(P1)))
        pairs_selected = P1[pairs_selected_index]
    else:  # two pair
        pairs_selected_index = np.random.choice(range(len(P2)))
        pairs_selected = P2[pairs_selected_index]

    # one unpair
    if len(unpairs_mt) == 1:
        unpairs_selected_index = np.random.choice(range(len(U1)))
        unpairs_selected = U1[unpairs_selected_index]
    else:  # two unpair
        unpairs_selected_index = np.random.choice(range(len(U2)))
        unpairs_selected = U2[unpairs_selected_index]

    nuc_list = list(seq)
    for pos_pair, pair in zip(pairs_mt, pairs_selected):
        nuc_list[pos_pair[0]] = pair[0]
        nuc_list[pos_pair[1]] = pair[1]
    for pos_unpair, unpair in zip(unpairs_mt, unpairs_selected):
        nuc_list[pos_unpair] = unpair
    return "".join(nuc_list)


# structured mutation

def mutate_structured_constrained(seq, pairs, v, v_list, T, constraints):
    v_list = [v / T for v in v_list]
    probs = np.exp(v_list) / sum(np.exp(v_list))
    index = np.random.choice(list(range(len(seq))), p=probs)
    pairs_mt = []
    unpairs_mt = []

    if index in pairs:
        i = min(index, pairs[index])
        j = max(index, pairs[index])
        pairs_mt.append((i, j))
        if j - 1 in pairs and pairs[j - 1] == i + 1:
            pairs_mt.append((pairs[j - 1], j - 1))
            if i + 2 not in pairs and j - 2 not in pairs:
                unpairs_mt.append(i + 2)
                unpairs_mt.append(j - 2)
        if i + 1 not in pairs and j - 1 not in pairs:
            unpairs_mt.append(i + 1)
            unpairs_mt.append(j - 1)
    else:
        unpairs_mt.append(index)
        if index - 1 in pairs and pairs[index - 1] > index:
            pairs_mt.append((index - 1, pairs[index - 1]))
            if pairs[index - 1] - 1 not in pairs:
                unpairs_mt.append(pairs[index - 1] - 1)
        elif index + 1 in pairs and pairs[index + 1] < index:
            pairs_mt.append((pairs[index + 1], index + 1))
            if pairs[index + 1] + 1 not in pairs:
                unpairs_mt.append(pairs[index + 1] + 1)

    assert len(pairs_mt) <= 2, pairs_mt
    assert len(unpairs_mt) <= 2, unpairs_mt
    pairs_selected = []
    unpairs_selected = ""
    # one pair
    if len(pairs_mt) == 1:
        (p_i, p_j) = pairs_mt[0]
        possible_pairs = pair_cache.get((constraints[p_i], constraints[p_j]), [])
        cache_key = (constraints[p_i], constraints[p_j])
        if cache_key not in pair_cache:
            print("no CACHE FOUND!")
            exit()
        pairs_selected_index = np.random.choice(range(len(possible_pairs)))
        pairs_selected = [possible_pairs[pairs_selected_index]]
    elif len(pairs_mt) == 2:  # two pair
        (p1_i, p1_j), (p2_i, p2_j) = pairs_mt[0], pairs_mt[1]
        possible_pairs1 = pair_cache.get((constraints[p1_i], constraints[p1_j]), [])
        possible_pairs2 = pair_cache.get((constraints[p2_i], constraints[p2_j]), [])
        if possible_pairs1 and possible_pairs2:
            valid_stacks = [[p1, p2] for p1 in possible_pairs1 for p2 in possible_pairs2]
        else:
            print("no valid stack can be made!")
            exit()

        if valid_stacks:
            random_index = np.random.choice(len(valid_stacks))
            # Use that index to select the full item (the stack)
            pairs_selected = valid_stacks[random_index]
        else:
            print("no valid stack!")
            exit()
    # one unpair
    if len(unpairs_mt) == 1:
        u1 = unpairs_mt[0]
        allowed_nucs = IUPAC_CODES.get(constraints[u1], [])
        unpairs_selected = unpairs_selected = np.random.choice(allowed_nucs)
    elif len(unpairs_mt) == 2:  # two unpair
        u1, u2 = unpairs_mt[0], unpairs_mt[1]
        allowed_nucs1 = IUPAC_CODES.get(constraints[u1], [])
        allowed_nucs2 = IUPAC_CODES.get(constraints[u2], [])
        if allowed_nucs1 and allowed_nucs2:
            valid_dinucs = [n1 + n2 for n1 in allowed_nucs1 for n2 in allowed_nucs2]
            unpairs_selected = np.random.choice(valid_dinucs)  # Result is a two-character string

    nuc_list = list(seq)
    for pos_pair, pair in zip(pairs_mt, pairs_selected):
        try:
            nuc_list[pos_pair[0]] = pair[0]
            nuc_list[pos_pair[1]] = pair[1]
        except Exception as e:
            print("size : ", pairs_mt, "  !! ", pairs_selected)
            print(pos_pair[0])
            print(pos_pair[1])
            print(pair[0])
            print(pair[1])

            exit()
    for pos_unpair, unpair_char in zip(unpairs_mt, unpairs_selected):
        nuc_list[pos_unpair] = unpair_char

    return "".join(nuc_list)


def samfeo(target, f, steps, k, t=1, check_mfe=True, sm=True, freq_print=FREQ_PRINT):
    start_time = time.time()
    global seed_np
    np.random.seed(seed_np)
    print(f'seed_np: {seed_np}')
    if sm:
        mutate = mutate_structured
    else:
        mutate = mutate_tradition
    print(f'steps: {steps}, t: {t}, k: {k}, structured mutation: {sm}, ensemble objective: {f.__name__}')

    # targeted initilization
    pairs = pairs_match(target)
    intial_list = init_k(target, pairs, k)
    history = set()
    k_best = []
    log = []
    dist_list = []
    mfe_list = []
    umfe_list = []
    count_umfe = 0
    ned_best = (1, None)
    ddg_best = (float('inf'), None)
    dist_best = (len(target), None)
    for p in intial_list:
        v_list, v, ss_list = f(p,
                               target)  # v_list: positional NED, v: objective value, ss_list: (multiple) MFE structures by subopt of ViennaRNA
        rna_struct = RNAStructure(seq=p, score=1 - v, v=v, v_list=v_list)
        rna_struct.dist = min([struct_dist(target, ss_subopt) for ss_subopt in ss_list])  # ss: secondary structure
        rna_struct.subcount = len(ss_list)
        k_best.append(rna_struct)
        history.add(rna_struct.seq)
        # record the best NED
        ned_p = np.mean(v_list)
        if ned_p <= ned_best[0]:
            ned_best = (ned_p, p)
        dist_p = min([struct_dist(target, ss_subopt) for ss_subopt in ss_list])
        if dist_p <= dist_best[0]:
            dist_best = (dist_p, p)
        ddg_p = delta_delta_energy(p, target, ss_list[0])
        if ddg_p <= ddg_best[0]:
            ddg_best = (ddg_p, p)

    # priority queue
    heapq.heapify(k_best)
    for i, rna_struct in enumerate(k_best):
        print(i, rna_struct)
        log.append(1 - rna_struct.score)
        if rna_struct.dist == 0:  # MFE solution
            mfe_list.append(rna_struct.seq)
        if rna_struct.dist == 0 and rna_struct.subcount == 1:  # UMFE solution
            dist_list.append(-2)
            umfe_list.append(rna_struct.seq)
            count_umfe += 1
        else:
            dist_list.append(rna_struct.dist)

    # log of lowest objective value at eachs iterations
    v_min = min(log)
    iter_min = 0
    log_min = [v_min]
    for i in range(steps):
        # sequence selection
        score_list = [rna_struct.score / t * 2 for rna_struct in k_best]  # objective values
        probs_boltzmann_1 = np.exp(score_list) / sum(np.exp(score_list))  # boltzmann distribution
        try:
            p = np.random.choice(k_best, p=probs_boltzmann_1)
        except Exception as e:
            print(e)
            p = np.random.choice(k_best)

        # position sampling and mutation
        seq_next = mutate(p.seq, pairs, p.v, p.v_list, t)
        num_repeat = 0
        while seq_next in history:
            num_repeat += 1
            if num_repeat > len(target) * MAX_REPEAT:
                break
            p = np.random.choice(k_best, p=probs_boltzmann_1)
            seq_next = mutate(p.seq, pairs, p.v, p.v_list, t)
        if num_repeat > len(target) * MAX_REPEAT:
            print(f'num_repeat: {num_repeat} > {len(target) * MAX_REPEAT}')
            break
        history.add(seq_next)

        # evaluation new sequence
        v_list_next, v_next, ss_list = f(seq_next, target)

        # mfe and umfe solutions as byproducts
        umfe = False
        if check_mfe:
            dist = min([struct_dist(target, ss_subopt) for ss_subopt in ss_list])
            if dist == 0:
                mfe_list.append(seq_next)
                if len(ss_list) == 1:
                    umfe = True
                    umfe_list.append(seq_next)
        else:
            dist = len(target)  # set a dummy dist
        if not umfe:
            dist_list.append(dist)
        else:
            dist_list.append(-2)
            count_umfe += 1

        # compare with best ned
        ned_next = np.mean(v_list_next)
        if ned_next <= ned_best[0]:
            ned_best = (ned_next, seq_next)
        dist_next = min([struct_dist(target, ss_subopt) for ss_subopt in ss_list])
        if dist_next <= dist_best[0]:
            dist_best = (dist_next, seq_next)
        ddg_next = delta_delta_energy(seq_next, target, ss_list[0])
        if ddg_next <= ddg_best[0]:
            ddg_best = (ddg_next, seq_next)

        # update priority queue(multi-frontier)
        rna_struct_next = RNAStructure(seq_next, 1 - v_next, v_next, v_list_next)

        if len(k_best) < k:
            heapq.heappush(k_best, rna_struct_next)
        elif rna_struct_next > k_best[0]:
            heapq.heappushpop(k_best, rna_struct_next)
        if v_next <= v_min:
            iter_min = i

        # update log
        v_min = min(v_min, v_next)
        log_min.append(v_min)
        log.append(v_next)
        assert len(dist_list) == len(log)

        # output information during iteration
        if (i + 1) % freq_print == 0:
            improve = v_min - log_min[-freq_print]
            if check_mfe:
                print(
                    f"iter: {i + 1: 5d}\t value: {v_min: .4f}\t mfe count: {len(mfe_list): 5d}\t umfe count: {count_umfe}\t best iter: {iter_min} improve: {improve:.2e}")
            else:
                print(f"iter: {i + 1: 5d}\t value: {v_min: .4f}\t best iter: {iter_min} improve: {improve:.2e}")

        # stop if convergency condition is satisfied
        if v_min < STOP or (len(log_min) > STAY and v_min - log_min[-STAY] > -1e-6):
            break
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    return k_best, log, mfe_list, umfe_list, dist_best, ned_best, ddg_best, elapsed_time


def samfeo_constrained(target, f, steps, k, t=1, check_mfe=True, sm=True, freq_print=FREQ_PRINT, constraints=None):
    pair_cache.clear()
    start_time = time.time()
    global seed_np
    np.random.seed(seed_np)
    print(f'seed_np: {seed_np}')
    if sm:
        mutate = mutate_structured_constrained
    else:
        mutate = mutate_tradition
    print(f'steps: {steps}, t: {t}, k: {k}, structured mutation: {sm}, ensemble objective: {f.__name__}')

    # targeted initilization
    pairs = pairs_match(target)
    intial_list = init_k_constrained(target, pairs, k, constraints)
    history = set()
    k_best = []
    log = []
    dist_list = []
    mfe_list = []
    umfe_list = []
    count_umfe = 0
    ned_best = (1, None)
    ddg_best = (float('inf'), None)
    dist_best = (len(target), None)
    for p in intial_list:
        v_list, v, ss_list = f(p,
                               target)  # v_list: positional NED, v: objective value, ss_list: (multiple) MFE structures by subopt of ViennaRNA
        rna_struct = RNAStructure(seq=p, score=1 - v, v=v, v_list=v_list)
        rna_struct.dist = min([struct_dist(target, ss_subopt) for ss_subopt in ss_list])  # ss: secondary structure
        rna_struct.subcount = len(ss_list)
        k_best.append(rna_struct)
        history.add(rna_struct.seq)
        # record the best NED
        ned_p = np.mean(v_list)
        if ned_p <= ned_best[0]:
            ned_best = (ned_p, p)
        dist_p = min([struct_dist(target, ss_subopt) for ss_subopt in ss_list])
        if dist_p <= dist_best[0]:
            dist_best = (dist_p, p)
        ddg_p = delta_delta_energy(p, target, ss_list[0])
        if ddg_p <= ddg_best[0]:
            ddg_best = (ddg_p, p)

    # priority queue
    heapq.heapify(k_best)
    for i, rna_struct in enumerate(k_best):
        print(i, rna_struct)
        log.append(1 - rna_struct.score)
        if rna_struct.dist == 0:  # MFE solution
            mfe_list.append(rna_struct.seq)
        if rna_struct.dist == 0 and rna_struct.subcount == 1:  # UMFE solution
            dist_list.append(-2)
            umfe_list.append(rna_struct.seq)
            count_umfe += 1
        else:
            dist_list.append(rna_struct.dist)

    # log of lowest objective value at eachs iterations
    v_min = min(log)
    iter_min = 0
    log_min = [v_min]
    for i in range(steps):
        # sequence selection
        score_list = [rna_struct.score / t * 2 for rna_struct in k_best]  # objective values
        probs_boltzmann_1 = np.exp(score_list) / sum(np.exp(score_list))  # boltzmann distribution
        try:
            p = np.random.choice(k_best, p=probs_boltzmann_1)
        except Exception as e:
            print(e)
            p = np.random.choice(k_best)

        # position sampling and mutation
        seq_next = mutate(p.seq, pairs, p.v, p.v_list, t, constraints)
        num_repeat = 0
        while seq_next in history:
            num_repeat += 1
            if num_repeat > len(target) * MAX_REPEAT:
                break
            p = np.random.choice(k_best, p=probs_boltzmann_1)
            seq_next = mutate(p.seq, pairs, p.v, p.v_list, t, constraints)
        if num_repeat > len(target) * MAX_REPEAT:
            print(f'num_repeat: {num_repeat} > {len(target) * MAX_REPEAT}')
            break
        history.add(seq_next)

        # evaluation new sequence
        v_list_next, v_next, ss_list = f(seq_next, target)

        # mfe and umfe solutions as byproducts
        umfe = False
        if check_mfe:
            dist = min([struct_dist(target, ss_subopt) for ss_subopt in ss_list])
            if dist == 0:
                mfe_list.append(seq_next)
                if len(ss_list) == 1:
                    umfe = True
                    umfe_list.append(seq_next)
        else:
            dist = len(target)  # set a dummy dist
        if not umfe:
            dist_list.append(dist)
        else:
            dist_list.append(-2)
            count_umfe += 1

        # compare with best ned
        ned_next = np.mean(v_list_next)
        if ned_next <= ned_best[0]:
            ned_best = (ned_next, seq_next)
        dist_next = min([struct_dist(target, ss_subopt) for ss_subopt in ss_list])
        if dist_next <= dist_best[0]:
            dist_best = (dist_next, seq_next)
        ddg_next = delta_delta_energy(seq_next, target, ss_list[0])
        if ddg_next <= ddg_best[0]:
            ddg_best = (ddg_next, seq_next)

        # update priority queue(multi-frontier)
        rna_struct_next = RNAStructure(seq_next, 1 - v_next, v_next, v_list_next)

        if len(k_best) < k:
            heapq.heappush(k_best, rna_struct_next)
        elif rna_struct_next > k_best[0]:
            heapq.heappushpop(k_best, rna_struct_next)
        if v_next <= v_min:
            iter_min = i

        # update log
        v_min = min(v_min, v_next)
        log_min.append(v_min)
        log.append(v_next)
        assert len(dist_list) == len(log)

        # output information during iteration
        if (i + 1) % freq_print == 0:
            improve = v_min - log_min[-freq_print]
            if check_mfe:
                print(
                    f"iter: {i + 1: 5d}\t value: {v_min: .4f}\t mfe count: {len(mfe_list): 5d}\t umfe count: {count_umfe}\t best iter: {iter_min} improve: {improve:.2e}")
            else:
                print(f"iter: {i + 1: 5d}\t value: {v_min: .4f}\t best iter: {iter_min} improve: {improve:.2e}")

        # stop if convergency condition is satisfied
        if v_min < STOP or (len(log_min) > STAY and v_min - log_min[-STAY] > -1e-6):
            break
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    return k_best, log, mfe_list, umfe_list, dist_best, ned_best, ddg_best, elapsed_time


def samfeo_para(args):
    target, f, steps, k, t, check_mfe, sm, freq_print = args
    return samfeo(target, f, steps, k, t, check_mfe, sm, freq_print)


# RNA design in batch
def design(path_txt, name , func, num_step, k, t, check_mfe, sm):
    targets = []
    with open(path_txt) as f:
        for line in f:
            targets.append(line.strip())
    data = []
    cols = (
        'puzzle_name', 'structure', 'rna', 'objective', 'mfe', 'dist_best', 'time', 'k_best', 'ned_best', 'ddg_best')
    if LOG:
        cols = ('puzzle_name', 'structure', 'rna', 'objective', 'mfe', 'dist_best', 'time', 'log', 'k_best', 'mfe_list',
                'umfe_list', 'ned_best', 'ddg_best')
    filename = f"{name}_{func.__name__}_t{t}_k{k}_step{num_step}_{name_pair}_{suffix}_mfe{check_mfe}_sm{sm}_time{int(time.time())}.csv"
    for i, target in enumerate(targets):
        puzzle_name = f"{name}_{i}"
        print(f'target structure {i}, {puzzle_name}:')
        print(target)
        start_time = time.time()
        k_best, log, mfe_list, umfe_list, dist_best, ned_best, ddg_best, elapsed_time = samfeo(target, func, num_step,
                                                                                               k=k, t=t,
                                                                                               check_mfe=check_mfe,
                                                                                               sm=sm)  # rna and ensemble defect
        finish_time = time.time()
        rna_best = max(k_best)
        seq = rna_best.seq
        obj = 1 - rna_best.score
        print('RNA sequence: ')
        print(seq)
        print('ensemble objective: ', obj)
        print(target)
        ss_mfe = mfe(seq)[0]
        dist = struct_dist(target, ss_mfe)
        print(ss_mfe)
        print(f'structure distance: {dist}')
        if LOG:
            data.append(
                [puzzle_name, target, seq, obj, ss_mfe, dist_best, elapsed_time, log, k_best, mfe_list, umfe_list,
                 ned_best, ddg_best])
        else:
            data.append([puzzle_name, target, seq, obj, ss_mfe, dist_best, elapsed_time, k_best, ned_best, ddg_best])
        # data.append([puzzle_name, target, seq, obj, ss_mfe, dist, finish_time-start_time, log, k_best, mfe_list, umfe_list, ned_best])
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(filename, index=False)


# RNA design with multiple processing
def design_para(path_txt, name, func, num_step, k, t, check_mfe, sm):
    from multiprocessing import Pool, cpu_count
    print('BATCH_SIZE:', BATCH_SIZE)
    print('WORKER_COUNT:', WORKER_COUNT)
    targets = []
    with open(path_txt) as f:
        for line in f:
            targets.append(line.strip())
    data = []
    cols = (
        'puzzle_name', 'structure', 'rna', 'objective', 'mfe', 'dist_best', 'time', 'k_best', 'ned_best', 'ddg_best')
    if LOG:
        cols = ('puzzle_name', 'structure', 'rna', 'objective', 'mfe', 'dist_best', 'time', 'log', 'k_best', 'mfe_list',
                'umfe_list', 'ned_best', 'ddg_best')
    filename = f"{name}_{func.__name__}_t{t}_k{k}_step{num_step}_{name_pair}_{suffix}_mfe{check_mfe}_sm{sm}_para_time{int(time.time())}.csv"
    for i_batch in range(0, len(targets), BATCH_SIZE):
        pool = Pool(WORKER_COUNT)
        args_map = []
        for j, target in enumerate(targets[i_batch: min(i_batch + BATCH_SIZE, len(targets))]):
            args_map.append((target, func, num_step, k, t, check_mfe, sm, FREQ_PRINT))
        print("args_map:")
        print(args_map)
        results_pool = pool.map(samfeo_para, args_map)
        pool.close()
        pool.join()
        for j, result in enumerate(results_pool):
            idx_puzzle = i_batch + j
            puzzle_name = f"{name}_{idx_puzzle}"
            target = targets[idx_puzzle]
            print(f'target structure {idx_puzzle}, {puzzle_name}:')
            print(target)
            k_best, log, mfe_list, umfe_list, dist_best, ned_best, ddg_best, elapsed_time = result

            rna_best = max(k_best)
            seq = rna_best.seq
            obj = 1 - rna_best.score
            print('RNA sequence: ')
            print(seq)
            print('ensemble objective: ', obj)
            print(target)
            ss_mfe = mfe(seq)[0]
            dist = struct_dist(target, ss_mfe)
            print(ss_mfe)
            print(f'structure distance: {dist}')
            if LOG:
                data.append(
                    [puzzle_name, target, seq, obj, ss_mfe, dist_best, elapsed_time, log, k_best, mfe_list, umfe_list,
                     ned_best, ddg_best])
            else:
                data.append(
                    [puzzle_name, target, seq, obj, ss_mfe, dist_best, elapsed_time, k_best, ned_best, ddg_best])
            df = pd.DataFrame(data, columns=cols)
            df.to_csv(filename, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--constrained", "-c", action='store_true',
                        help="Activate constrained design mode. Expects a second input line for IUPAC constraints.")
    parser.add_argument("--path", '-p', type=str, default='')
    parser.add_argument("--object", '-o', type=str, default='pd')
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--t", type=float, default=1)
    parser.add_argument("--step", type=int, default=5000)
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--init", type=str, default='cg')
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--nomfe", action='store_true')
    parser.add_argument("--nosm", action='store_true')
    parser.add_argument("--bp", action='store_true')
    parser.add_argument("--nolog", action='store_true')
    parser.add_argument("--online", action='store_true')
    parser.add_argument("--para", action='store_true')
    parser.add_argument("--worker_count", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=20)

    args = parser.parse_args()
    print('args:')
    print(args)
    global name_pair, stop, seed_np
    name_pair = args.init
    name_input = args.path.split("/")[-1].split('.')[0]
    if args.object == 'ned':  # normalized ensemble defect
        f_obj = position_ed_ned_mfe
    elif args.object == 'pd':  # probability defect
        f_obj = position_ed_pd_mfe
    else:
        raise ValueError('the objective in not correct!')
    LOG = not args.nolog
    if args.online:
        seed_np = 2020
        np.random.seed(seed_np)
        # Check if the constrained flag is used
        if args.constrained:
            # --- BEHAVIOR 1: CONSTRAINED MODE ---
            print("Online CONSTRAINED mode. Enter structure/constraint pairs continuously.")
            for target_line, constraint_line in zip(sys.stdin, sys.stdin):
                target = target_line.strip()
                constraints = constraint_line.strip()

                if not target: continue

                if len(target) != len(constraints):
                    print(
                        "\nError: The structure and constraint strings must have the same length. Skipping this pair.")
                    continue

                print("\n--- Processing New CONSTRAINED Puzzle ---")
                print(f"Structure: {target}")
                print(f"Constraint: {constraints}")

                k_best, log, mfe_list, umfe_list, dist_best, ned_best, ddg_best, elapsed_time = samfeo_constrained(
                    target, f_obj,
                    args.step,
                    k=args.k,
                    t=args.t,
                    check_mfe=not args.nomfe,
                    sm=not args.nosm,
                    constraints=constraints)

                # --- Print Results ---
                rna_best = max(k_best)
                seq = rna_best.seq
                obj = 1 - rna_best.score
                print('RNA sequence: ')
                print(seq)
                print('ensemble objective: ', obj)
                print(target)
                ss_mfe = mfe(seq)[0]
                dist = struct_dist(target, ss_mfe)
                print(ss_mfe)
                print(f'structure distance: {dist}')
                print(f'count of mfe solutsion: {len(mfe_list)}')
                print(f'count of umfe solutions: {len(umfe_list)}')
                print(k_best)
                kbest_list = []
                for rna_struct in k_best:
                    obj = 'prob' if args.object == 'pd' else 'ned'
                    # print(f'seq: {rna_struct.seq}, {obj}: {rna_struct.score}')
                    kbest_list.append({'seq': rna_struct.seq, obj: rna_struct.score})
                print(' mfe samples:', mfe_list[-10:])
                print('umfe samples:', umfe_list[-10:])
                print('kbest:', k_best)
                print('ned_best:', ned_best)
                results = {'kbest': kbest_list, 'mfe': mfe_list, 'umfe': umfe_list, 'ned_best': ned_best}
                filename = "_".join(
                    ["puzzle", target.replace('(', '[').replace(')', ']'), "seed", str(seed_np)]) + ".json"
                with open(filename, 'w') as f:
                    json.dump(results, f)
                print(f"full results are saved in the file: {filename}")
                print('\nRNA sequence:', rna_best.seq)
                print('---------------------------\n')
                print('---------------------------\n')




                detailed_results = analyze_and_format_k_best(k_best, target)

                # 3. Create a pandas DataFrame and save it to a CSV file
                df = pd.DataFrame(detailed_results)

                # Create a safe filename from the target structure
                safe_filename = target.replace('.', '_').replace('(', '[').replace(')', ']')
                output_filename = f"(constrained) NEEEDDD222 k_best_analysis_{safe_filename}.csv"
                pairs = pairs_match(target)
                mini, maxi = calculate_gc_bounds(target, constraints, pairs)
                print("miniiiiiiiiiiiiiiiiiiiiiiiiiiii , ", mini, maxi)
                df.to_csv(output_filename, index=False)
                print(f"\nDetailed analysis for the top {len(k_best)} sequences saved to: {output_filename}")

        else:
            # --- BEHAVIOR 2: UNCONSTRAINED MODE ---
            print("Online UNCONSTRAINED mode. Enter structures continuously.")
            print("Press Ctrl+Z then Enter (Windows) or Ctrl+D (Linux/macOS) to exit.")

            # Use a simple loop to read one line at a time
            for line in sys.stdin:
                target = line.strip()
                constraints = None  # Constraints are explicitly None in this mode

                if not target: continue

                print("\n--- Processing New UNCONSTRAINED Puzzle ---")
                print(f"Structure: {target}")

                k_best, log, mfe_list, umfe_list, dist_best, ned_best, ddg_best, elapsed_time = samfeo(
                    target, f_obj,
                    args.step,
                    k=args.k,
                    t=args.t,
                    check_mfe=not args.nomfe,
                    sm=not args.nosm)

                # --- Print Results ---
                rna_best = max(k_best)
                seq = rna_best.seq
                obj = 1 - rna_best.score
                print('RNA sequence: ')
                print(seq)
                print('ensemble objective: ', obj)
                print(target)
                ss_mfe = mfe(seq)[0]
                dist = struct_dist(target, ss_mfe)
                print(ss_mfe)
                print(f'structure distance: {dist}')
                print(f'count of mfe solutsion: {len(mfe_list)}')
                print(f'count of umfe solutions: {len(umfe_list)}')
                print(k_best)
                kbest_list = []
                for rna_struct in k_best:
                    obj = 'prob' if args.object == 'pd' else 'ned'
                    # print(f'seq: {rna_struct.seq}, {obj}: {rna_struct.score}')
                    kbest_list.append({'seq': rna_struct.seq, obj: rna_struct.score})
                print(' mfe samples:', mfe_list[-10:])
                print('umfe samples:', umfe_list[-10:])
                print('kbest:', k_best)
                print('ned_best:', ned_best)
                results = {'kbest': kbest_list, 'mfe': mfe_list, 'umfe': umfe_list, 'ned_best': ned_best}
                filename = "_".join(
                    ["puzzle", target.replace('(', '[').replace(')', ']'), "seed", str(seed_np)]) + ".json"
                with open(filename, 'w') as f:
                    json.dump(results, f)
                print(f"full results are saved in the file: {filename}")
                print('\nRNA sequence:', rna_best.seq)
                print('---------------------------\n')
            exit(0)

    for i in range(args.repeat):
        seed_np = 2020 + (i + args.start) * 2021
        np.random.seed(seed_np)
        suffix = f"{i + args.start}"
        if args.para:
            WORKER_COUNT = args.worker_count
            BATCH_SIZE = args.batch_size
            design_para(args.path, name_input, f_obj, args.step, k=args.k, t=args.t, check_mfe=not args.nomfe,
                        sm=not args.nosm)
        else:
            design(args.path, name_input, f_obj, args.step, k=args.k, t=args.t, check_mfe=not args.nomfe,
                   sm=not args.nosm)
