import json
import os
import sys
from collections import Counter
from pathlib import Path

import dpctl
import numpy

import dpnp

from . import config as warn_config


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
    EVENT_PREFIX = "DPNP_WARNING_EVENT - "

    def __init__(self):
        self.enabled = warn_config.infra_warnings_enable
        self.directory = warn_config.infra_warnings_directory
        self.events_artifact = warn_config.infra_warnings_events_artifact
        self.summary_artifact = warn_config.infra_warnings_summary_artifact

        self._counts = Counter()
        self._warnings = {}
        self._totals = Counter()
        self._env = {}

        self._events_fp = None
        self._events_file = None
        self._summary_file = None

    def _log_stdout(self, message: str) -> None:
        try:
            sys.stderr.write(message.rstrip("\n") + "\n")
            sys.stderr.flush()
        except Exception:
            pass

    def pytest_configure(self, _config):
        if not self.enabled:
            return

        self._env.update(
            {
                "numpy_version": getattr(numpy, "__version__", "unknown"),
                "numpy_path": getattr(numpy, "__file__", "unknown"),
                "dpnp_version": getattr(dpnp, "__version__", "unknown"),
                "dpnp_path": getattr(dpnp, "__file__", "unknown"),
                "dpctl_version": getattr(dpctl, "__version__", "unknown"),
                "dpctl_path": getattr(dpctl, "__file__", "unknown"),
                "job": os.getenv("JOB_NAME", "unknown"),
                "build_number": os.getenv("BUILD_NUMBER", "unknown"),
                "git_sha": os.getenv("GIT_COMMIT", "unknown"),
            }
        )

        if self.directory:
            try:
                p = Path(self.directory).expanduser().resolve()
                if p.exists() and not p.is_dir():
                    raise ValueError(f"{p} exists and is not a directory")

                p.mkdir(parents=True, exist_ok=True)

                if (
                    not self.events_artifact
                    or Path(self.events_artifact).name != self.events_artifact
                ):
                    raise ValueError(
                        f"Invalid events artifact filename: {self.events_artifact}"
                    )

                if (
                    not self.summary_artifact
                    or Path(self.summary_artifact).name != self.summary_artifact
                ):
                    raise ValueError(
                        f"Invalid summary artifact filename: {self.summary_artifact}"
                    )

                self._events_file = p / self.events_artifact
                self._events_fp = self._events_file.open(
                    mode="w", encoding="utf-8", buffering=1, newline="\n"
                )
                self._summary_file = p / self.summary_artifact
            except Exception as exc:
                self._close_events_fp()
                self._log_stdout(
                    "DPNP infra warnings plugin: artifacts disabled "
                    f"(failed to initialize directory/files): {exc}"
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

        if key not in self._warnings:
            self._warnings[key] = {
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

        self._log_stdout(f"{self.EVENT_PREFIX} {_json_dumps_one_line(event)}")

    def pytest_terminal_summary(self, terminalreporter, exitstatus, _config):
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
                dict(self._warnings[k], count=c)
                for k, c in self._counts.most_common(50)
                if k in self._warnings
            ],
        }

        if self._summary_file:
            try:
                with open(self._summary_file, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, sort_keys=True)
                terminalreporter.write_line(
                    f"DPNP infrastructure warnings summary written to: {self._summary_file}"
                )
            except Exception as exc:
                terminalreporter.write_line(
                    f"Failed to write DPNP infrastructure warnings summary to: {self._summary_file}. Error: {exc}"
                )

        self._close_events_fp()
        terminalreporter.write_line(self.SUMMARY_BEGIN)
        terminalreporter.write_line(_json_dumps_one_line(summary))
        terminalreporter.write_line(self.SUMMARY_END)

    def pytest_unconfigure(self, _config):
        self._close_events_fp()

    def _close_events_fp(self):
        if self._events_fp is None:
            return
        try:
            self._events_fp.close()
        finally:
            self._events_fp = None


def register_infra_warnings_plugin_if_enabled(config) -> None:
    """Register infra warnings plugin if enabled via env var."""

    if not warn_config.infra_warnings_enable:
        return

    plugin_name = "dpnp-infra-warnings"
    if config.pluginmanager.get_plugin(plugin_name) is not None:
        return

    config.pluginmanager.register(DpnpInfraWarningsPlugin(), plugin_name)
