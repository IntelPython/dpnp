import os


def _getenv_bool(name):
    # unset, empty, "0", "false", "no" and "off" mean disabled
    value = os.getenv(name, "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


all_int_types = _getenv_bool("DPNP_TEST_ALL_INT_TYPES")
float16_types = _getenv_bool("DPNP_TEST_FLOAT_16")
complex_types = _getenv_bool("DPNP_TEST_COMPLEX_TYPES")
bool_types = _getenv_bool("DPNP_TEST_BOOL_TYPES")
skip_tensor_tests = _getenv_bool("SKIP_TENSOR_TESTS")
queue_gpu = _getenv_bool("DPNP_QUEUE_GPU")


infra_warnings_enable = _getenv_bool("DPNP_INFRA_WARNINGS_ENABLE")
infra_warnings_directory = os.getenv("DPNP_INFRA_WARNINGS_DIRECTORY", None)
infra_warnings_events_artifact = os.getenv(
    "DPNP_INFRA_WARNINGS_EVENTS_ARTIFACT",
    "dpnp_infra_warnings_events.jsonl",
)
infra_warnings_summary_artifact = os.getenv(
    "DPNP_INFRA_WARNINGS_SUMMARY_ARTIFACT",
    "dpnp_infra_warnings_summary.json",
)
