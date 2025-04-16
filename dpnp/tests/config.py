import os

all_int_types = bool(os.getenv("DPNP_TEST_ALL_INT_TYPES", 0))
float16_types = bool(os.getenv("DPNP_TEST_FLOAT_16", 0))
complex_types = bool(os.getenv("DPNP_TEST_COMPLEX_TYPES", 0))
bool_types = bool(os.getenv("DPNP_TEST_BOOL_TYPES", 0))
