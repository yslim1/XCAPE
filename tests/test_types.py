from xcape.utils.types import RelaxResult, copy_relax_result


def test_relax_result_defaults(nacl_structure):
    result = RelaxResult(
        structure=nacl_structure,
        energy=-3.5,
        status='converged',
        t_rel=1.2,
    )
    assert result.origin_type == 'random'
    assert result.parent == 'none'
    assert result.t_gen is None
    assert result.r_factor is None
    assert result.wasserstein is None


def test_copy_relax_result_deep_copies_structure(nacl_structure):
    original = RelaxResult(
        structure=nacl_structure,
        energy=-3.5,
        status='converged',
        t_rel=1.2,
        r_factor=0.05,
        wasserstein=0.2,
        gen=3,
        pop=7,
        origin_type='kept_best',
        parent='2_1',
    )
    copied = copy_relax_result(original)

    assert copied is not original
    assert copied.structure is not original.structure
    assert copied.structure == original.structure
    assert copied.energy == original.energy
    assert copied.status == original.status
    assert copied.t_rel == original.t_rel
    assert copied.r_factor == original.r_factor
    assert copied.wasserstein == original.wasserstein
    assert copied.gen == original.gen
    assert copied.pop == original.pop
    assert copied.origin_type == original.origin_type
    assert copied.parent == original.parent

    # Mutating the copy's structure must not affect the original.
    copied.structure.apply_strain(0.1)
    assert copied.structure != original.structure
