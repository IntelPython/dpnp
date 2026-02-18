import os

all_int_types = bool(os.getenv("DPNP_TEST_ALL_INT_TYPES", 0))
float16_types = bool(os.getenv("DPNP_TEST_FLOAT_16", 0))
complex_types = bool(os.getenv("DPNP_TEST_COMPLEX_TYPES", 0))
bool_types = bool(os.getenv("DPNP_TEST_BOOL_TYPES", 0))


infra_warnings_enable = bool(os.getenv("DPNP_INFRA_WARNINGS_ENABLE", 0))
infra_warnings_directory = os.getenv("DPNP_INFRA_WARNINGS_DIRECTORY", None)
infra_warnings_events_artifact = os.getenv(
    "DPNP_INFRA_WARNINGS_EVENTS_ARTIFACT",
    "dpnp_infra_warnings_events.jsonl",
)
infra_warnings_summary_artifact = os.getenv(
    "DPNP_INFRA_WARNINGS_SUMMARY_ARTIFACT",
    "dpnp_infra_warnings_summary.json",
)
