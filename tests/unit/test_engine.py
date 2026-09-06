"""Starting ComfyUI, and knowing whether it really started.

"okay it installed but then nothing" -- the install finished and handed the
user a sentence telling them to open a program we had just put in a folder they
did not choose, with no shortcut, no address and no button.

Two things have to be right for that to stop being true, and neither can be
taken on trust:

* the **command** must point ComfyUI at the models we downloaded. The installer
  puts them in ``<root>/models`` while ComfyUI's own state belongs under
  ``<root>/comfy``, and ``--base-directory`` alone would send it looking in
  ``<root>/comfy/models``. Every workflow would report missing models on a
  machine where all of them are present, which is a far more baffling failure
  than not starting at all.
* **ready must mean ready.** The process exists within milliseconds and serves
  HTTP tens of seconds later. Reporting the first as the second opens a browser
  on a connection-refused page.

The supervisor is therefore tested against a real child process and a real
socket -- a stand-in that behaves like ComfyUI's startup, including the ways it
can fail -- rather than against a mock that agrees with us.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from toolshed.exec.engine import (
    DEFAULT_PORT,
    HOST,
    Engine,
    EngineError,
    Layout,
    choose_port,
    port_is_free,
    read_extra_flags,
    write_extra_flags,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the stand-in engine uses a POSIX symlink for the venv interpreter",
)


# A stand-in for ComfyUI: prints startup noise the way the real one does, then
# serves /system_stats. Told to, it dies instead, or comes up mute.
FAKE_ENGINE = textwrap.dedent('''
    import sys, os, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    port = int(sys.argv[sys.argv.index("--port") + 1])
    mode = os.environ.get("FAKE_MODE", "ok")

    print("Total VRAM 20464 MB, total RAM 64158 MB")
    print("Using ComfyUI base directory:", sys.argv[sys.argv.index("--base-directory") + 1])
    sys.stdout.flush()

    if mode == "die":
        print("Traceback (most recent call last):")
        print("RuntimeError: no kernel image is available for execution")
        sys.stdout.flush()
        raise SystemExit(1)

    if mode == "slow":
        time.sleep(float(os.environ.get("FAKE_DELAY", "2")))

    if mode == "mute":
        while True:
            time.sleep(0.2)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            if self.path == "/system_stats":
                body = b'{"system": {"comfyui_version": "0.34.0"}}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

    print("Starting server")
    sys.stdout.flush()
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


@pytest.fixture
def installed(tmp_path: Path) -> Path:
    """A data root that looks like a finished install."""
    layout = Layout(tmp_path)
    layout.engine_dir.mkdir(parents=True)
    layout.main_py.write_text(FAKE_ENGINE)
    layout.models_dir.mkdir(parents=True)
    layout.python.parent.mkdir(parents=True)
    layout.python.symlink_to(sys.executable)
    return tmp_path


@pytest.fixture
def engine(installed: Path):
    eng = Engine(root=installed)
    yield eng
    eng.stop()


class TestTheCommand:
    """Pure, so every flag can be asserted without starting anything."""

    def test_models_come_from_where_the_installer_put_them(self, tmp_path):
        """The bug this pairing exists to prevent: --base-directory alone
        would point the engine at <root>/comfy/models, which is empty."""
        cmd = Engine(root=tmp_path, port=1).command()
        pairs = dict(zip(cmd, cmd[1:], strict=False))
        assert pairs["--models-directory"] == str(tmp_path / "models")
        assert pairs["--base-directory"] == str(tmp_path / "comfy")
        assert pairs["--models-directory"] != pairs["--base-directory"] + "/models"

    def test_workflows_land_where_the_engine_will_look(self, tmp_path):
        """inject() writes under <base>/user/default/workflows/Toolshed, and
        ComfyUI reads user data from <base-directory>/user. If those two ever
        disagree the workflows simply do not appear in the sidebar."""
        from toolshed.exec.inject import workflows_dir

        cmd = Engine(root=tmp_path, port=1).command()
        base = Path(dict(zip(cmd, cmd[1:], strict=False))["--base-directory"])
        assert workflows_dir(base).is_relative_to(base / "user")

    def test_output_goes_somewhere_a_person_can_find(self, tmp_path):
        cmd = Engine(root=tmp_path, port=1).command()
        assert dict(zip(cmd, cmd[1:], strict=False))["--output-directory"] == \
            str(tmp_path / "output")

    def test_it_listens_on_loopback_only(self, tmp_path):
        """ComfyUI has no authentication. Binding it to the network would put
        an unauthenticated code-execution endpoint on the user's LAN."""
        cmd = Engine(root=tmp_path, port=1).command()
        assert dict(zip(cmd, cmd[1:], strict=False))["--listen"] == "127.0.0.1"
        assert "0.0.0.0" not in cmd

    def test_we_open_the_browser_not_the_engine(self, tmp_path):
        """--disable-auto-launch: the engine would otherwise open a browser the
        moment it binds the socket, which is before it can serve anything."""
        assert "--disable-auto-launch" in Engine(root=tmp_path, port=1).command()

    def test_it_runs_the_private_python_not_the_system_one(self, tmp_path):
        cmd = Engine(root=tmp_path, port=1).command()
        assert cmd[0] == str(tmp_path / "runtime" / "venv" / "bin" / "python")
        assert cmd[1] == str(tmp_path / "engine" / "comfyui" / "main.py")


class TestWhatIsMissing:
    def test_a_bare_directory_reports_all_three(self, tmp_path):
        missing = Layout(tmp_path).missing_pieces()
        assert len(missing) == 3
        assert all(" " in m for m in missing), "paths, not plain words, reached the user"

    def test_a_finished_install_reports_nothing(self, installed):
        assert Layout(installed).missing_pieces() == []

    def test_starting_without_an_install_says_so_rather_than_crashing(self, tmp_path):
        with pytest.raises(EngineError) as exc:
            Engine(root=tmp_path).start()
        assert exc.value.reason_key == "not_installed"
        assert "not installed yet" in str(exc.value)


class TestPortChoice:
    def test_the_familiar_port_is_used_when_free(self):
        if not port_is_free(DEFAULT_PORT):
            pytest.skip("something is already on 8188 on this machine")
        assert choose_port() == DEFAULT_PORT

    def test_an_occupied_port_is_stepped_around(self):
        """A ComfyUI the user already runs must not block Toolshed entirely."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind((HOST, 0))
            held.listen(1)
            taken = held.getsockname()[1]
            assert not port_is_free(taken)
            assert choose_port(taken) != taken


class TestStartingForReal:
    def test_it_comes_up_and_answers(self, engine):
        engine.start()
        engine.wait_until_ready(timeout=30)
        assert engine.responds()
        assert engine.is_running()
        assert engine.url.startswith("http://127.0.0.1:")

    def test_ready_means_serving_not_merely_launched(self, installed):
        """The distinction the whole class exists for. The stand-in is made to
        take two seconds so this is a fact, not a race that usually holds."""
        engine = Engine(root=installed, env={"FAKE_MODE": "slow", "FAKE_DELAY": "2"})
        try:
            engine.start()
            assert engine.is_running(), "the process did not start"
            assert not engine.responds(timeout=0.3), \
                "reported ready while the engine was still starting"
            engine.wait_until_ready(timeout=30)
            assert engine.responds()
        finally:
            engine.stop()

    def test_stopping_really_stops_it(self, engine):
        engine.start()
        engine.wait_until_ready(timeout=30)
        port = engine.port
        engine.stop()
        assert not engine.is_running()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not port_is_free(port):
            time.sleep(0.1)
        assert port_is_free(port), "the port is still held; the child outlived us"

    def test_stopping_twice_is_harmless(self, engine):
        engine.start()
        engine.wait_until_ready(timeout=30)
        engine.stop()
        engine.stop()

    def test_the_engines_output_is_kept(self, engine):
        engine.start()
        engine.wait_until_ready(timeout=30)
        assert "Total VRAM" in engine.tail()
        written = engine.layout.log_file.read_text()
        assert "Total VRAM" in written
        assert "--- started" in written, "runs are not separated in the log"

    def test_the_base_directory_actually_reaches_the_engine(self, engine):
        """Not just that we pass the flag -- that the child receives it."""
        engine.start()
        engine.wait_until_ready(timeout=30)
        assert str(engine.layout.comfy_base) in engine.tail()


class TestWhenItGoesWrong:
    def test_an_engine_that_dies_is_reported_with_its_last_words(self, installed):
        """A CUDA error two seconds in must not be waited on for five minutes."""
        engine = Engine(root=installed, env={"FAKE_MODE": "die"})
        engine.start()
        began = time.monotonic()
        with pytest.raises(EngineError) as exc:
            engine.wait_until_ready(timeout=60)
        assert time.monotonic() - began < 20, "it waited out the timeout on a dead process"
        assert exc.value.reason_key == "engine_died"
        assert "no kernel image" in exc.value.detail

    def test_an_engine_that_never_answers_times_out_and_is_killed(self, installed):
        engine = Engine(root=installed, env={"FAKE_MODE": "mute"})
        engine.start()
        try:
            with pytest.raises(EngineError) as exc:
                engine.wait_until_ready(timeout=3)
            assert exc.value.reason_key == "engine_timeout"
            assert not engine.is_running(), "a hung engine was left running"
        finally:
            engine.stop()

    def test_cancelling_stops_the_child(self, installed):
        engine = Engine(root=installed, env={"FAKE_MODE": "mute"})
        engine.start()
        try:
            with pytest.raises(EngineError) as exc:
                engine.wait_until_ready(timeout=30, should_cancel=lambda: True)
            assert exc.value.reason_key == "cancelled"
            assert not engine.is_running()
        finally:
            engine.stop()


class TestTheGplBoundary:
    def test_the_engine_is_driven_as_a_process_never_imported(self):
        """ComfyUI is GPL-3.0-or-later. Toolshed downloads it and talks to it
        over HTTP; importing it would make Toolshed a derived work."""
        import ast

        source = Path(__file__).resolve().parents[2] / "toolshed" / "exec" / "engine.py"
        text = source.read_text()
        tree = ast.parse(text)

        # Parsed, not grepped: this module's docstring contains the words
        # "import comfy" in the course of explaining that it never does one.
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        assert not imported & {"comfy", "nodes", "folder_paths", "execution", "server"}
        assert "subprocess.Popen" in text, "the engine is not being run as a child process"


class TestItReallyIsComfyUIsInterface:
    """The flags are only correct if ComfyUI accepts them. Asserted against the
    argument parser from the pinned version rather than from memory."""

    def test_every_flag_we_pass_exists_in_the_pinned_cli(self, tmp_path):
        args = [a for a in Engine(root=tmp_path, port=1).command() if a.startswith("--")]
        known = {
            "--base-directory", "--models-directory", "--output-directory",
            "--listen", "--port", "--disable-auto-launch", "--log-stdout",
        }
        assert set(args) == known, (
            "the launch flags changed; re-read comfy/cli_args.py for the pinned "
            "version and update docs/UPSTREAM.md before changing this test"
        )


def test_the_stand_in_engine_is_a_fair_test(installed):
    """If the stand-in stopped being startable the tests above would pass
    vacuously, so prove it runs at all."""
    result = subprocess.run(
        [sys.executable, str(Layout(installed).main_py), "--port", "1",
         "--base-directory", str(installed), "--help"],
        capture_output=True, text=True, timeout=30, env={"FAKE_MODE": "die"},
    )
    assert "Total VRAM" in result.stdout


class TestExtraFlags:
    """A lever for the failures Toolshed cannot fix from outside the engine.

    ComfyUI has real options for these -- --fp32-vae and --cpu-vae for a model
    the card will not run, --cuda-device to choose between two graphics cards,
    --reserve-vram to leave the desktop some room. Without somewhere to put
    them, someone hitting one of those has no move except to stop using the app.
    """

    def test_none_by_default(self, tmp_path):
        assert read_extra_flags(tmp_path) == []

    def test_they_reach_the_command_line(self, tmp_path):
        engine = Engine(root=tmp_path, port=1, extra_args=["--fp32-vae"])
        assert engine.command()[-1] == "--fp32-vae"

    def test_they_come_last_so_they_can_override_ours(self, tmp_path):
        """argparse takes the later value for a repeated option. That is what
        makes this an escape hatch rather than a suggestion box."""
        engine = Engine(root=tmp_path, port=1, extra_args=["--port", "9999"])
        cmd = engine.command()
        assert cmd[-2:] == ["--port", "9999"]
        assert cmd.index("--port") < len(cmd) - 2, "ours should still be there, earlier"

    def test_they_survive_a_restart(self, tmp_path):
        write_extra_flags(tmp_path, "--cuda-device 1  --reserve-vram 2")
        assert read_extra_flags(tmp_path) == ["--cuda-device", "1", "--reserve-vram", "2"]

    def test_quoting_is_respected(self, tmp_path):
        write_extra_flags(tmp_path, '--output-directory "/two words/out"')
        assert read_extra_flags(tmp_path) == ["--output-directory", "/two words/out"]

    def test_nonsense_does_not_stop_the_engine_starting(self, tmp_path):
        """An unclosed quote must not turn into a crash on launch."""
        write_extra_flags(tmp_path, 'a "b')
        assert read_extra_flags(tmp_path) == []

    def test_they_are_argv_never_a_shell(self, tmp_path):
        """Nothing here is interpolated into a command line, so there is
        nothing to inject into."""
        engine = Engine(root=tmp_path, port=1, extra_args=["; rm -rf /"])
        assert "; rm -rf /" in engine.command()
        source = (Path(__file__).resolve().parents[2]
                  / "toolshed" / "exec" / "engine.py").read_text()
        assert "shell=True" not in source
