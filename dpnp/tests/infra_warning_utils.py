import json
import os
import sys

from collections import Counter

import dpctl
import numpy

import dpnp


def _env_check(var_name: str, *, default: bool = False) -> bool:
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _origin_from_filename(filename: str) -> str:
    file = (filename or "").replace("\\", "/")
    if "/dpnp/" in file or file.startswith("dpnp/"):
        return "dpnp"
    if "/numpy/" in file or file.startswith("numpy/"):
        return "numpy"
    if "/dpctl/" in file or file.startswith("dpctl/"):
        return "dpctl"
    return "third_party"


def _json_dumps_one_line(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


class DpnpInfraWarningsPlugin:
    """Pytest custom plugin that records pytest-captured warnings.

    It only records what pytest already captures (via pytest_warning_recorded).
    Does not change warnings filters.

    Env vars:
    - DPNP_INFRA_WARNINGS_ENABLE=1 (enables the plugin)
    - DPNP_INFRA_WARNINGS_DIRECTORY=<dir> (writes artifacts)
    - DPNP_INFRA_WARNINGS_EVENTS_ARTIFACT (optional filename)
    - DPNP_INFRA_WARNINGS_SUMMARY_ARTIFACT (optional filename)
    """

    SUMMARY_BEGIN = "DPNP_WARNINGS_SUMMARY_BEGIN"
    SUMMARY_END = "DPNP_WARNINGS_SUMMARY_END"
    EVENT_PREFIX = "DPNP_WARNING_EVENT "

    def __init__(self):
        self.enabled = _env_check("DPNP_INFRA_WARNINGS_ENABLE", default=False)
        self.directory = os.getenv("DPNP_INFRA_WARNINGS_DIRECTORY", None)
        self.events_artifact = os.getenv(
            "DPNP_INFRA_WARNINGS_EVENTS_ARTIFACT", "dpnp_infra_warnings_events.jsonl"
        )
        self.summary_artifact = os.getenv(
            "DPNP_INFRA_WARNINGS_SUMMARY_ARTIFACT", "dpnp_infra_warnings_summary.json"
        )

        self.print_events = self.enabled

        self._counts = Counter()
        self._examples = {}
        self._totals = Counter()
        self._env = {}

        self._events_fp = None
        self._events_file = None

    def pytest_configure(self, config):
        if not self.enabled:
            return

        try:
            numpy_version = numpy.__version__
            numpy_path = getattr(numpy, "__file__", "unknown")
            dpnp_version = dpnp.__version__
            dpnp_path = getattr(dpnp, "__file__", "unknown")
            dpctl_version = dpctl.__version__
            dpctl_path = getattr(dpctl, "__file__", "unknown")
        except Exception:
            numpy_version = "unknown"
            numpy_path = "unknown"
            dpnp_version = "unknown"
            dpnp_path = "unknown"
            dpctl_version = "unknown"
            dpctl_path = "unknown"

        if self.directory:
            os.makedirs(self.directory, exist_ok=True)
            self._events_file = os.path.join(self.directory, self.events_artifact)
            self._events_fp = open(
                self._events_file,
                "w",
                encoding="utf-8",
                buffering=1,
                newline="\n",
            )

        self._env.update(
            {
                "numpy_version": numpy_version,
                "numpy_path": numpy_path,
                "dpnp_version": dpnp_version,
                "dpnp_path": dpnp_path,
                "dpctl_version": dpctl_version,
                "dpctl_path": dpctl_path,
                "job": os.getenv("JOB_NAME", "unknown"),
                "build_number": os.getenv("BUILD_NUMBER", "unknown"),
                "git_sha": os.getenv("GIT_COMMIT", "unknown"),
                "events_file": self._events_file,
            }
        )

    def pytest_warning_recorded(self, warning_message, when, nodeid, location):
        if not self.enabled:
            return

        category = getattr(
            getattr(warning_message, "category", None),
            "__name__",
            str(getattr(warning_message, "category", "Warning")),
        )
        message = str(getattr(warning_message, "message", warning_message))

        filename = getattr(warning_message, "filename", None) or (
            location[0] if location and len(location) > 0 else None
        )
        lineno = getattr(warning_message, "lineno", None) or (
            location[1] if location and len(location) > 1 else None
        )
        func = location[2] if location and len(location) > 2 else None

        origin = _origin_from_filename(filename or "")
        key = f"{category}||{origin}||{message}"
        self._counts[key] += 1
        self._totals[f"category::{category}"] += 1
        self._totals[f"origin::{origin}"] += 1
        self._totals[f"phase::{when}"] += 1

        if key not in self._examples:
            self._examples[key] = {
                "category": category,
                "origin": origin,
                "when": when,
                "nodeid": nodeid,
                "filename": filename,
                "lineno": lineno,
                "function": func,
                "message": message,
            }

        event = {
            "when": when,
            "nodeid": nodeid,
            "category": category,
            "origin": origin,
            "message": message,
            "filename": filename,
            "lineno": lineno,
            "function": func,
        }

        if self._events_fp is not None:
            try:
                self._events_fp.write(_json_dumps_one_line(event) + "\n")
            except Exception:
                pass

        if self.print_events:
            try:
                sys.stderr.write(self.EVENT_PREFIX + _json_dumps_one_line(event) + "\n")
                sys.stderr.flush()
            except Exception:
                pass

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        if not self.enabled:
            return

        summary = {
            "schema_version": "1.0",
            "exit_status": exitstatus,
            "environment": dict(self._env),
            "total_warning_events": int(sum(self._counts.values())),
            "unique_warning_types": int(len(self._counts)),
            "totals": dict(self._totals),
            "top_unique_warnings": [
                dict(self._examples[k], count=c)
                for k, c in self._counts.most_common(50)
                if k in self._examples
            ],
        }

        if self.directory:
            output_file = os.path.join(self.directory, self.summary_artifact)
            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, sort_keys=True)
                terminalreporter.write_line(
                    f"DPNP infrastructure warnings summary written to: {output_file}"
                )
            except Exception as exc:
                terminalreporter.write_line(
                    f"Failed to write DPNP infrastructure warnings summary to: {output_file}. Error: {exc}"
                )

        self._close_events_fp()

        terminalreporter.write_line(self.SUMMARY_BEGIN)
        terminalreporter.write_line(_json_dumps_one_line(summary))
        terminalreporter.write_line(self.SUMMARY_END)

    def pytest_unconfigure(self, config):
        self._close_events_fp()

    def _close_events_fp(self):
        if self._events_fp is None:
            return
        try:
            self._events_fp.close()
        except Exception:
            pass
        self._events_fp = None


def register_infra_warnings_plugin_if_enabled(config) -> None:
    """Register infra warnings plugin if enabled via env var."""

    if not _env_check("DPNP_INFRA_WARNINGS_ENABLE"):
        return

    plugin_name = "dpnp-infra-warnings"
    if config.pluginmanager.get_plugin(plugin_name) is not None:
        return

    config.pluginmanager.register(DpnpInfraWarningsPlugin(), plugin_name)
