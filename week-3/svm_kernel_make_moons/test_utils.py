from kernel_svm_make_moons_cross_validation import scale_features # or just import scale_features if in same file

def test_scaled_mean_is_zero():
    result = scale_features([1, 2, 3, 4, 5])
    mean = sum(result) / len(result)
    assert abs(mean) < 1e-6  # test that mean ≈ 0
