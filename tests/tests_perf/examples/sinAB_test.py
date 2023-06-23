import time


def cos_2_args(executor, size, test_type):
    """sin(A + B) = sin A cos B + cos A sin B"""

    start_time = time.perf_counter()
    input_A = executor.arange(size, dtype=test_type)
    input_B = executor.arange(size, dtype=test_type)
    end_time = time.perf_counter()
    memalloc_time = end_time - start_time

    start_time = time.perf_counter()

    sin_A = executor.sin(input_A)
    cos_B = executor.cos(input_B)
    sincosA = sin_A * cos_B

    cos_A = executor.cos(input_A)
    sin_B = executor.sin(input_B)
    sincosB = cos_A * sin_B

    result = sincosA + sincosB

    end_time = time.perf_counter()
    calculation_time = end_time - start_time

    print(
        f"memalloc_time={memalloc_time}, calculation_time={calculation_time}, executor={executor}"
    )

    return result


if __name__ == "__main__":
    size = 33554432  # 16777216

    import dpnp

    cos_2_args(dpnp, size, dpnp.float64)

    import numpy

    cos_2_args(numpy, size, numpy.float64)
