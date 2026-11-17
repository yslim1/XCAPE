"""Mutation operators for XCAPE structure generation."""

from typing import Any, Optional, Dict, List, Tuple, Callable, Sequence

from functools import lru_cache
from types import SimpleNamespace

import random
import numpy as np
import copy

from pyxtal.wyckoff_site import atom_site
from pyxtal.tolerance import Tol_matrix


def _normalize_mults(mults: Sequence[int]) -> Tuple[int]:
    return tuple(sorted(set(int(m) for m in mults)))


@lru_cache(maxsize=128)
def _all_multisets_sum_to(target: int, mults: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    Return all multisets (as nondecreasing tuples) of elements in `mults` that sum to `target`.
    Example: target=12, mults=(6,12) -> [(6,6), (12,)]
    """
    mults = tuple(sorted(mults))

    # recursion with nondecreasing constraint to avoid permutations
    results: List[Tuple[int, ...]] = []

    def rec(remaining: int, start_idx: int, acc: List[int]):
        if remaining == 0:
            results.append(tuple(acc))
            return
        if remaining < 0:
            return
        for i in range(start_idx, len(mults)):
            m = mults[i]
            if m > remaining:
                break
            acc.append(m)
            rec(remaining - m, i, acc)  # allow reuse, keep nondecreasing
            acc.pop()

    rec(target, 0, [])
    return results


def sample_multiset(
    target: int,
    allowed_mults: Sequence[int],
    rng: random.Random = random,
    weight_fn: Optional[Callable[[Tuple[int, ...]], float]] = None,
) -> Tuple[int, ...]:
    """
    Sample one multiset of multiplicities summing to target.

    weight_fn: optional function to bias selection.
      - larger weight => more likely
      - if None: uniform over all valid multisets
    """
    mults = _normalize_mults(allowed_mults)
    all_sets = _all_multisets_sum_to(int(target), mults)

    if not all_sets:
        raise ValueError(f"No multiset decomposition for target={target} using {mults}")

    if weight_fn is None:
        return rng.choice(all_sets)

    weights = [max(0.0, float(weight_fn(ms))) for ms in all_sets]
    s = sum(weights)
    if s <= 0:
        # fallback uniform if weights all zero
        return rng.choice(all_sets)

    # weighted choice without numpy
    r = rng.random() * s
    cum = 0.0
    for ms, w in zip(all_sets, weights):
        cum += w
        if r <= cum:
            return ms
    return all_sets[-1]

class Mutator:
    """
    Mutation operator collection for crystal structure prediction.

    Supported moves:
    - swap between Wyckoff sites with the same multiplicity
    - split a site into two sites with matching total multiplicity
    - merge two same-species sites into one larger-multiplicity site
    - swap species between sites with the same multiplicity
    - reconfigure all sites belonging to one species
    - resample one or multiple movable positions

    These moves preserve composition and reject candidates that violate the
    configured short-distance tolerance checks.
    """
    def __init__(
        self,
        swap_same_mult_ratio: float = 0.25,
        split_site_ratio: float = 0.1,
        merge_site_ratio: float = 0.1,
        swap_diff_elem_ratio: float = 0.15,
        site_reconfig_ratio: float = 0.2,
        resample_pos_ratio: float = 0.1,
        multi_resample_pos_ratio: float = 0.1,
        tolerance_matrix: object = None,
        max_attempts: int = 100,
    ):
        """
        Initialize the mutator with move ratios and parameters.
        Parameters:
        - swap_same_mult_ratio: Ratio for Wyckoff swap with same multiplicity move.
        - split_site_ratio: Ratio for split Wyckoff-site move.
        - merge_site_ratio: Ratio for merge Wyckoff-site move.
        - swap_diff_elem_ratio: Ratio for Wyckoff swap with different elements move.
        - site_reconfig_ratio: Ratio for random Wyckoff-site reconfiguration move.
        - resample_pos_ratio: Ratio for resampling atomic positions move.
        - multi_resample_pos_ratio: Ratio for multi-site resampling move.
        - tolerance_matrix: Tolerance matrix object for distance checks.
        - max_attempts: Maximum attempts to perform a valid move.
        """
        self.swap_same_mult_ratio = swap_same_mult_ratio
        self.split_site_ratio = split_site_ratio
        self.merge_site_ratio = merge_site_ratio
        self.swap_diff_elem_ratio = swap_diff_elem_ratio
        self.site_reconfig_ratio = site_reconfig_ratio
        self.resample_pos_ratio = resample_pos_ratio
        self.multi_resample_pos_ratio = multi_resample_pos_ratio

        if abs(sum([
            self.swap_same_mult_ratio,
            self.split_site_ratio,
            self.merge_site_ratio,
            self.swap_diff_elem_ratio,
            self.site_reconfig_ratio,
            self.resample_pos_ratio,
            self.multi_resample_pos_ratio,
        ])-1.0) > 1e-6:
            raise ValueError("The sum of all move ratios must be 1.0")
        
        if tolerance_matrix is None:
            self.tolerance_matrix = Tol_matrix()
        else:
            self.tolerance_matrix = tolerance_matrix
        
        self.max_attempts = max_attempts
        
    
    def _group_wps_by_multiplicity(self, spg: object) -> Dict[int, List[object]]:
        """
        Group Wyckoff positions by multiplicity.
        
        Input:
            spg: Space group object
        Output:
            wps_by_mult: Dictionary mapping multiplicity to list of Wyckoff positions
        """
        
        wps_by_mult: Dict[int, List[object]] = {}
        for wp in spg.Wyckoff_positions:
            wps_by_mult.setdefault(wp.multiplicity, []).append(wp)
        return wps_by_mult
    

    def _used_fixed_labels(self, structure: object) -> List[str]:
        """
        Collect labels of already-used dof=0 Wyckoff positions.
        
        Input:
            structure: pyxtal structure object
        Output:
            used_fixed_labels: List of labels of used dof=0 Wyckoff positions
        """
        
        labels = set()
        for site in structure.atom_sites:
            wp = site.wp
            if wp.get_dof() == 0:
                labels.add(wp.get_label())
        return list(labels)

    def _make_atom_site(self, wp: Any, specie: str) -> atom_site:
        return atom_site(wp=wp, coordinate=list(np.random.random(3)), specie=specie)
    
    
    def _check_short_distance_using_wp(self, skip_indices, structure: object, target_atom_site: object) -> bool:
        """
        Check distance validity against other sites after changing one site.
        
        Input:
            structure: pyxtal structure object
            atom_site: atom site object with updated Wyckoff position
        Output:
            is_short_distances: Boolean indicating if there are short distances
        """

        raw_atom_sites = structure.atom_sites.copy()
        lattice = structure.lattice.matrix

        is_valid = True
        for i, raw_atom_site in enumerate(raw_atom_sites):
            if i in skip_indices:
                continue

            is_valid = raw_atom_site.check_with_ws2(
                ws2=target_atom_site,
                lattice=lattice,
                tm=self.tolerance_matrix,
            )
            
            if not is_valid:
                return is_valid
            
        return is_valid
    

    def _check_short_distance_self_wp(self, structure: object, target_atom_site: object) -> bool:
        """
        Check self-distance validity within the target Wyckoff orbit.
        
        Input:
            structure: pyxtal structure object
            atom_site: atom site object with updated Wyckoff position
        Output:
            is_short_distances: Boolean indicating if there are short distances
        """

        wp = target_atom_site.wp
        pt = target_atom_site.position
        cell = structure.lattice.matrix

        specie = target_atom_site.specie
        tol = self.tolerance_matrix.get_tol(specie, specie)

        is_valid = wp.distance_check(pt, cell, tol)

        return is_valid


    def _commit_candidate_sites(
        self,
        structure: object,
        remove_indices: Sequence[int],
        new_sites: Sequence[object],
    ) -> bool:
        """
        Validate candidate sites against a trial structure and commit atom sites
        only when the full replacement is valid.
        """
        trial_atom_sites = copy.deepcopy(structure.atom_sites)
        trial_structure = SimpleNamespace(
            atom_sites=trial_atom_sites,
            lattice=structure.lattice,
        )

        for idx in sorted(remove_indices, reverse=True):
            trial_structure.atom_sites.pop(idx)

        for new_site in new_sites:
            if not self._check_short_distance_self_wp(trial_structure, new_site):
                return False
            if not self._check_short_distance_using_wp([], trial_structure, new_site):
                return False
            trial_structure.atom_sites.append(new_site)

        structure.atom_sites = trial_structure.atom_sites
        return True


    def _get_movable_site_indices(self, structure: object) -> List[int]:
        """Return indices of atom sites with positional degrees of freedom."""
        return [idx for idx, site in enumerate(structure.atom_sites) if site.dof > 0]


    def _mutation_choices(self) -> List[Tuple[str, float, Callable]]:
        """Return the configured mutation dispatch table."""
        return [
            ("swap_same_mult", self.swap_same_mult_ratio, self.swap_wyckoff_site_same_multiplicity),
            ("split", self.split_site_ratio, self.split_wyckoff_site),
            ("merge", self.merge_site_ratio, self.merge_wyckoff_site),
            ("swap_diff_elem", self.swap_diff_elem_ratio, self.swap_wyckoff_site_different_elements),
            ("reconfig", self.site_reconfig_ratio, self.wyckoff_site_random_reconfiguration),
            ("resample", self.resample_pos_ratio, self.resampling_atomic_positions),
            ("multi_resample", self.multi_resample_pos_ratio, self.multi_resampling_atomic_positions),
        ]


    def swap_wyckoff_site_same_multiplicity(self, structure: Any, max_attempts: int = 100) -> Tuple[Any, bool]:
        """
        Swap to another Wyckoff site with the same multiplicity.

        Caveat:
        - already-occupied dof=0 Wyckoff labels cannot be reused
        
        Input:
            structure: pyxtal structure object
        Output:
            structure: pyxtal structure object with swapped wyckoff sites
        """

        is_changed = False

        spg = structure.group

        # 1) Build: multiplicity -> list of Wyckoff positions
        wps_by_mult = self._group_wps_by_multiplicity(spg)
        
        # 2) Collect labels of already-used dof=0 Wyckoff positions
        used_fixed_labels = self._used_fixed_labels(structure)

        # 3) For each site, list alternative Wyckoff positions with the same multiplicity
        candidate_wps_by_site: Dict[int, List[object]] = {}
        changeable_site_indices: List[int] = []

        for i, site in enumerate(structure.atom_sites):
            curr_wp = site.wp
            mult = curr_wp.multiplicity

            candidates: List[object] = []
            for wp in wps_by_mult.get(mult, []):
                if wp.get_label() == curr_wp.get_label():
                    continue # keep same wp out
                if wp.get_label() in used_fixed_labels:
                    continue # keep dof=0 used wp out
                candidates.append(wp)
            
            candidate_wps_by_site[i] = candidates
            if candidates:
                changeable_site_indices.append(i)

        if not changeable_site_indices: # no available site to change
            return structure, is_changed
        
        # 4) Choose one site and one new Wyckoff position randomly, then update
        site_idx = random.choice(changeable_site_indices)
        old_site = structure.atom_sites[site_idx]
    
        new_wp = random.choice(candidate_wps_by_site[site_idx])

        attempts = 0
        while attempts < max_attempts:
            attempts += 1

            new_atom_site = self._make_atom_site(wp=new_wp, specie=old_site.specie)

            if self._commit_candidate_sites(structure, [site_idx], [new_atom_site]):
                is_changed = True
                return structure, is_changed

        return structure, is_changed


    def split_wyckoff_site(self, structure: Any, max_attempts: int = 100) -> Tuple[Any, bool]:
        """
        Split one Wyckoff site into two sites with the same total multiplicity.

        Policy:
        - Choose ONE existing atom_site (orbit) with multiplicity M
        - Replace it with TWO new sites of the same element with multiplicities m1 and m2 such that m1 + m2 = m
        - Avoid using dof=0 labels already used when possible

        Input:
            structure: pyxtal structure object
        Output:
            structure: pyxtal structure object with split wyckoff sites
        """

        is_changed = False

        spg = structure.group
        wps_by_mult = self._group_wps_by_multiplicity(spg)
        used_fixed_labels = self._used_fixed_labels(structure)

        # 1) Pick candidate sites that can be split
        #  i.e., their multiplicity m can be written as m1+m2 where both exist in spg.
        split_candidates: List[Tuple[int, int, int]] = [] # (site_index, m1, m2)
        for i, site in enumerate(structure.atom_sites):
            m = site.wp.multiplicity
            # Enumerated splits m1+m2=m with available multiplicites
            for m1 in wps_by_mult.keys():
                m2 = m - m1
                if m2 <= 0:
                    continue
                if m2 not in wps_by_mult:
                    continue
                if m1 > m2: # keep m1 <= m2 to avoid duplications
                    continue
                split_candidates.append((i, m1, m2))

        if not split_candidates:
            return structure, is_changed
        
        # 2) Choose specific Wyckoff positions for m1 and m2
        site_idx, m1, m2 = random.choice(split_candidates)
        old_site = structure.atom_sites[site_idx]

        freed_fixed_labels = set()
        if old_site.wp.get_dof() == 0:
            freed_fixed_labels.add(old_site.wp.get_label())
        blocked_fixed_labels = set(used_fixed_labels) - freed_fixed_labels

        def _candidate_wps(mult: int) -> List[object]:
            candidates = []
            for wp in wps_by_mult.get(mult, []):
                if wp.get_label() in blocked_fixed_labels:
                    continue
                candidates.append(wp)
            return candidates

        candidate_pairs: List[Tuple[object, object]] = []
        for wp1 in _candidate_wps(m1):
            for wp2 in _candidate_wps(m2):
                if wp1.get_dof() == 0 and wp2.get_dof() == 0 and wp1.get_label() == wp2.get_label():
                    continue
                candidate_pairs.append((wp1, wp2))

        if not candidate_pairs:
            return structure, is_changed

        wp1, wp2 = random.choice(candidate_pairs)

        attempts = 0
        while attempts < max_attempts:
            attempts += 1

            site_1 = self._make_atom_site(wp=wp1, specie=old_site.specie)
            site_2 = self._make_atom_site(wp=wp2, specie=old_site.specie)

            if self._commit_candidate_sites(structure, [site_idx], [site_1, site_2]):
                is_changed = True
                return structure, is_changed

        return structure, is_changed


    def merge_wyckoff_site(self, structure: Any, max_attempts: int = 100) -> Tuple[Any, bool]:
        """
        Merge two same-species Wyckoff sites into one larger site.

        Policy:
        - Pick TWO atom sites with the same element
        - Let their multiplicities be m1, m2; target multiplicity m = m1 + m2
        - Replace the two sites with ONE new site of multiplicity m
        - Avoid using dof=0 labels already used, BUT allow labels freed by removing the two sites
        """

        is_changed = False

        spg = structure.group
        wps_by_mult = self._group_wps_by_multiplicity(spg)

        # Collect used fixed labels (dof=0) in the current structure
        used_fixed_labels = set(self._used_fixed_labels(structure))

        # 1) Build merge candidates: (i, j, target_m)
        # - same element
        # - target multiplicity exists in spg
        merge_candidates: List[Tuple[int, int, int]] = [] # (site_index_1, site_index_2, target_m)
        sites = structure.atom_sites

        for i in range(len(sites)):
            for j in range(i+1, len(sites)):
                si, sj = sites[i], sites[j]

                # Same element check
                if si.specie != sj.specie:
                    continue

                m1 = si.wp.multiplicity
                m2 = sj.wp.multiplicity
                target_m = m1 + m2

                if target_m not in wps_by_mult.keys():
                    continue

                merge_candidates.append((i, j, target_m))

        if not merge_candidates: # no available candidates to merge
            return structure, is_changed
        
        # 2) Pick a candidate pair
        i, j, target_m = random.choice(merge_candidates)
        site_i = sites[i]
        site_j = sites[j]

        # These labels might be freed after removing sites i and j
        freed_fixed_labels = set()
        if site_i.wp.get_dof() == 0:
            freed_fixed_labels.add(site_i.wp.get_label())
        if site_j.wp.get_dof() == 0:
            freed_fixed_labels.add(site_j.wp.get_label())

        # used_fixed_labels minus freed ones
        blocked_fixed_labels = used_fixed_labels - freed_fixed_labels

        # 3) Pick target wp for target_m
        candidates = []
        for wp in wps_by_mult.get(target_m, []):
            # avoid dof=0 label collisions with other already-used fixed labels
            if wp.get_label() in blocked_fixed_labels:
                continue
            candidates.append(wp)

        if not candidates: # no available target wp
            return structure, is_changed
        
        new_wp = random.choice(candidates)

        # 4) Create merged site based on one old site's dict (same element)
        attempts = 0
        while attempts < max_attempts:
            attempts += 1

            merged_site = self._make_atom_site(wp=new_wp, specie=site_i.specie)

            if self._commit_candidate_sites(structure, [i, j], [merged_site]):
                is_changed = True
                return structure, is_changed

        return structure, is_changed


    def swap_wyckoff_site_different_elements(self, structure: Any, max_attempts: int = 100) -> Tuple[Any, bool]:
        """
        Swap species between two sites with the same multiplicity.

        Policy:
        - Pick TWO atom sites i != j with different species
        - Swap only species (keep wp + positions unchanged)
        - They must have the same multiplicity to preserve composition
        - This preserves composition and orbit integrity.
        """
        
        is_changed = False
        sites = structure.atom_sites

        # 1) Find candidate pairs (i, j) with different species
        diff_pairs = []
        for i in range(len(sites)):
            for j in range(i+1, len(sites)):
                if (sites[i].specie != sites[j].specie) and (sites[i].wp.multiplicity == sites[j].wp.multiplicity):
                    diff_pairs.append((i, j))

        if not diff_pairs: # no available pairs to swap
            return structure, is_changed

        attempts = 0
        while attempts < max_attempts:
            attempts += 1

            # Swapping species is deterministic for a given pair (unlike the
            # position-resampling moves, which re-roll a coordinate each try),
            # so re-sample the pair every attempt -- otherwise every retry would
            # repeat the identical failing swap.
            i, j = random.choice(diff_pairs)

            # Check distance validity after swapping species
            sites[i].specie, sites[j].specie = sites[j].specie, sites[i].specie

            # 3) Check self-distance validity
            if not self._check_short_distance_self_wp(structure, sites[i]):
                sites[i].specie, sites[j].specie = sites[j].specie, sites[i].specie
                continue
            if not self._check_short_distance_self_wp(structure, sites[j]):
                sites[i].specie, sites[j].specie = sites[j].specie, sites[i].specie
                continue

            # 4) Check distance validity
            if not self._check_short_distance_using_wp([i], structure, sites[i]):
                sites[i].specie, sites[j].specie = sites[j].specie, sites[i].specie
                continue
            if not self._check_short_distance_using_wp([j], structure, sites[j]):
                sites[i].specie, sites[j].specie = sites[j].specie, sites[i].specie
                continue
            
            is_changed = True

            return structure, is_changed
        
        return structure, is_changed
    

    def wyckoff_site_random_reconfiguration(self, structure: Any, max_attempts: int = 100) -> Tuple[Any, bool]:
        """
        Randomly reassign all Wyckoff sites for one species.

        Policy:
        - For a selected element, reassign all of its sites using available
          Wyckoff positions with matching multiplicities.
        Input:
            structure: pyxtal structure object
        Output:
            structure: pyxtal structure object with reconfigured wyckoff sites
        """
        is_changed = False

        # --- choose specie to reconfigure ---
        # Sort the deduplicated species so the pick is reproducible under a
        # fixed seed: plain set/frozenset iteration order for strings depends on
        # PYTHONHASHSEED, which would make random.choice diverge run-to-run even
        # when random.seed() is set. key=str keeps it robust if specie is not a
        # bare string.
        species = sorted({s.specie for s in structure.atom_sites}, key=str)
        if not species:
            return structure, is_changed
        change_specie = random.choice(species)

        kept_sites = [s for s in structure.atom_sites if s.specie != change_specie]
        old_sites = [s for s in structure.atom_sites if s.specie == change_specie]
        if not old_sites:
            return structure, is_changed

        total_multiplicity = sum(s.wp.multiplicity for s in old_sites)

        # --- blocked dof=0 labels: keep those used by other species, free those used by change_specie ---
        used_fixed = set(self._used_fixed_labels(structure))
        freed_fixed = {
            s.wp.get_label() for s in old_sites
            if s.wp.get_dof() == 0
        }
        blocked_fixed = used_fixed - freed_fixed

        # --- build usable wps from full space group ---
        useable_wps: Dict[int, List[object]] = {}
        for wp in structure.group.Wyckoff_positions:
            if wp.get_dof() == 0 and (wp.get_label() in blocked_fixed):
                continue
            useable_wps.setdefault(wp.multiplicity, []).append(wp)

        if not useable_wps:
            return structure, is_changed

        allowed_mults = list(useable_wps.keys())

        # helper: incremental short-distance check against an explicit list of sites
        def _check_against_sites(sites: List[object], new_site: object) -> bool:
            lattice = structure.lattice.matrix
            for s in sites:
                if not s.check_with_ws2(ws2=new_site, lattice=lattice, tm=self.tolerance_matrix):
                    return False
            return True

        # --- try multiple times to create a full new configuration ---
        for _ in range(max_attempts):
            # reset on each outer attempt
            new_sites: List[object] = []
            base_sites = kept_sites[:]  # fixed background

            # sample multiplicity multiset
            try:
                ms = sample_multiset(total_multiplicity, allowed_mults=allowed_mults)
            except ValueError:
                break
            
            used_fixed_in_new = set()
            ok = True
            for m in ms:
                found = False
                for _ in range(self.max_attempts):
                    wp = random.choice(useable_wps[m])

                    if wp.get_dof() == 0 and wp.get_label() in used_fixed_in_new:
                        continue
                    
                    cand = self._make_atom_site(wp=wp, specie=change_specie)

                    if not self._check_short_distance_self_wp(structure, cand):
                        continue

                    # check vs kept + already-accepted new sites
                    if not _check_against_sites(base_sites, cand):
                        continue
                    if not _check_against_sites(new_sites, cand):
                        continue

                    if wp.get_dof() == 0:
                        used_fixed_in_new.add(wp.get_label())

                    new_sites.append(cand)
                    found = True
                    break

                if not found:
                    ok = False
                    break

            if not ok:
                continue

            # success: atomic commit
            structure.atom_sites = kept_sites + new_sites
            
            is_changed = True

            return structure, is_changed

        return structure, is_changed

    def resampling_atomic_positions(self, structure: Any, max_attempts: int = 100) -> Tuple[Any, bool]:
        """
        Resample one movable atomic position within fractional coordinates.
        
        Input:
            structure: pyxtal structure object
        Output:
            structure: pyxtal structure object with resampled atomic positions
        """

        is_changed = False

        dof_indices = self._get_movable_site_indices(structure)

        if len(dof_indices) == 0: # no available site to resample
            return structure, is_changed

        rand_idx = random.choice(dof_indices)

        # resampling
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            orig_pos = list(structure.atom_sites[rand_idx].position)
            random_pos = list(map(np.float64, np.random.rand(1,3)[0]))
            structure.atom_sites[rand_idx].update(random_pos)

            # check distance validity
            if not self._check_short_distance_self_wp(structure, structure.atom_sites[rand_idx]):
                structure.atom_sites[rand_idx].update(orig_pos)
                continue

            if not self._check_short_distance_using_wp([rand_idx], structure, structure.atom_sites[rand_idx]):
                structure.atom_sites[rand_idx].update(orig_pos)
                continue
            
            is_changed = True

            return structure, is_changed

        return structure, is_changed


    def multi_resampling_atomic_positions(self, structure: Any, max_attempts: int = 100) -> Tuple[Any, bool]:
        """
        Resample positions of multiple movable sites simultaneously.

        Policy:
        - Randomly choose a subset of movable sites.
        - Resample only the selected fractional positions together.
        - Commit the update only if the full trial configuration remains valid.

        Input:
            structure: pyxtal structure object
        Output:
            structure: pyxtal structure object with resampled atomic positions
        """

        is_changed = False

        dof_indices = self._get_movable_site_indices(structure)

        if len(dof_indices) < 2: # need at least 2 movable sites for multi-resampling, if not enough, same as single resampling
            return structure, is_changed

        for _ in range(max_attempts):
            subset_size = random.randint(2, len(dof_indices))
            selected_indices = sorted(random.sample(dof_indices, k=subset_size))

            trial_atom_sites = copy.deepcopy(structure.atom_sites)
            trial_structure = SimpleNamespace(
                atom_sites=trial_atom_sites,
                lattice=structure.lattice,
            )

            for idx in selected_indices:
                random_pos = list(map(np.float64, np.random.rand(3)))
                trial_structure.atom_sites[idx].update(random_pos)

            is_valid = True
            for idx in selected_indices:
                trial_site = trial_structure.atom_sites[idx]
                if not self._check_short_distance_self_wp(trial_structure, trial_site):
                    is_valid = False
                    break
                if not self._check_short_distance_using_wp([idx], trial_structure, trial_site):
                    is_valid = False
                    break

            if not is_valid:
                continue

            for idx in selected_indices:
                structure.atom_sites[idx].update(trial_structure.atom_sites[idx].position)

            is_changed = True
            return structure, is_changed

        return structure, is_changed

    
    def mutate(self, structure: Any) -> Tuple[Any, bool, str]:
        """
        Perform one mutation attempt based on the configured move ratios.

        This method samples a single move according to the configured ratios
        and tries that move once (with the move's internal ``max_attempts``).
        If the sampled move fails, the caller is responsible for retrying by
        calling ``mutate`` again.

        Input:
            structure: pyxtal structure object
        Output:
            structure: pyxtal structure object after attempting a move
        """

        mutate_choices = self._mutation_choices()

        rand_val = random.random()
        cumulative_ratio = 0.0

        for mutate_name, ratio, mutate_func in mutate_choices:
            cumulative_ratio += ratio
            if rand_val <= cumulative_ratio:

                orig_sites = copy.deepcopy(structure.atom_sites)

                structure, is_changed = mutate_func(structure, max_attempts=self.max_attempts)
                if not is_changed:
                    # The sampled move failed after its own internal attempts.
                    # Higher-level retry logic lives in the caller.
                    structure.atom_sites = orig_sites
                    return structure, False, mutate_name
                    
                # successful move
                return structure, True, mutate_name

        # Shouldn't happen if ratios sum to 1.0, but safe fallback
        return structure, False, "none"
