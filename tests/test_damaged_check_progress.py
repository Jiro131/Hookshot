"""The damaged-texture check after an unpack says how far it is: nup.damaged_game_files reports each file it opened, and
make_game's file_progress turns that into the builder's @@progress (at most 4 a second, and the last) and a plain line at each
tenth. Without them the build showed "reading it again is skipped" and then nothing for minutes. Game-free."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nx2"))
sys.path.insert(0, str(ROOT / "third_party/a64-kit/tools"))
import nup  # noqa: E402

spec = importlib.util.spec_from_file_location("make_game_damaged_progress", ROOT / "nx2/make_game.py")
make_game = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_game)


class DamagedFilesProgressTests(unittest.TestCase):
    def test_each_file_reported_and_the_damaged_one_found(self):
        with tempfile.TemporaryDirectory(prefix="hookshot-damaged-") as romfs:
            os.makedirs(os.path.join(romfs, "Model"))
            for name in ("a.bfres.zs", "c.bfres.zs"):
                with open(os.path.join(romfs, "Model", name), "wb") as f:
                    f.write(nup.ZSTD + b"rest")
            damaged = os.path.join(romfs, "Model", "b.bfres.zs")
            with open(damaged, "wb") as f:
                f.write(b"\0" * 8)
            with open(os.path.join(romfs, "Model", "not_zs.txt"), "wb") as f:
                f.write(b"\0" * 8)
            calls = []
            self.assertEqual(nup.damaged_game_files(romfs, lambda done, total: calls.append((done, total))), [damaged])
            self.assertEqual(calls, [(1, 3), (2, 3), (3, 3)])
            self.assertEqual(nup.damaged_game_files(romfs), [damaged])   # (no progress: as before)

    def test_many_files_in_the_listed_order(self):
        # (the files are opened several at a time: the damaged ones and the progress still come in the listed order,
        # past a slice of 4096 and with more files than threads, and a file that cannot be read still stops the check)
        with tempfile.TemporaryDirectory(prefix="hookshot-damaged-") as romfs:
            listed, damaged = [], []
            for folder in ("Model", "Pack/Actor", "UI"):
                os.makedirs(os.path.join(romfs, folder))
                for n in range(1500):
                    path = os.path.normpath(os.path.join(romfs, folder, "%04d.bfres.zs" % n))
                    bad = n % 97 == 3
                    with open(path, "wb") as f:
                        f.write(b"\0" * 8 if bad else nup.ZSTD + b"rest")
                    listed.append(path)
                    if bad:
                        damaged.append(path)
            calls = []
            self.assertEqual(nup.damaged_game_files(romfs, lambda done, total: calls.append((done, total))), damaged)
            self.assertEqual(calls, [(n, len(listed)) for n in range(1, len(listed) + 1)])
            unreadable = listed[2000]
            real_frame = nup.still_in_nup_frame
            def frame(f, head):
                if os.path.normpath(f.name).endswith(os.path.relpath(unreadable, romfs)):
                    raise PermissionError(13, "in use", unreadable)
                return real_frame(f, head)
            with mock.patch.object(nup, "still_in_nup_frame", side_effect=frame):
                with self.assertRaises(PermissionError):
                    nup.damaged_game_files(romfs)

    def test_files_opened_several_at_a_time(self):
        # (the point of the change: the first two opens wait for each other, which one file at a time never gets past)
        with tempfile.TemporaryDirectory(prefix="hookshot-damaged-") as romfs:
            os.makedirs(os.path.join(romfs, "Model"))
            for n in range(8):
                with open(os.path.join(romfs, "Model", "%d.bfres.zs" % n), "wb") as f:
                    f.write(nup.ZSTD + b"rest")
            both = threading.Barrier(2, timeout=5)
            opened = []
            lock = threading.Lock()
            def opening(path, *args, **kwargs):
                with lock:
                    opened.append(path)
                    first_two = len(opened) <= 2
                if first_two:
                    both.wait()   # (BrokenBarrierError after 5 s when nothing else is opening)
                return open(path, *args, **kwargs)
            with mock.patch.object(nup, "open", side_effect=opening, create=True):
                self.assertEqual(nup.damaged_game_files(romfs), [])
            self.assertEqual(len(opened), 8)

    def test_file_progress_lines(self):
        t = [0.0]
        def clock():
            return t[0]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            progress = make_game.file_progress("checked", clock)
            for done in range(1, 101):
                t[0] = done / 10   # (10 files a second: every third or so may send @@progress)
                progress(done, 100)
        lines = out.getvalue().splitlines()
        bars = [line for line in lines if line.startswith("@@progress ")]
        said = [line for line in lines if not line.startswith("@@")]
        self.assertEqual(bars[-1], "@@progress 1.000")
        self.assertLessEqual(len(bars), 10 * 4 + 2)   # (10 s of checking: at most 4 a second, the first and the last)
        self.assertGreater(len(bars), 5)
        self.assertEqual([float(b.split()[1]) for b in bars], sorted(float(b.split()[1]) for b in bars))
        self.assertEqual(len(said), 10)
        self.assertEqual(said[3], "checked 40 of 100 files (40%, 4 s)")
        self.assertEqual(said[-1], "checked 100 of 100 files (100%, 10 s)")

    def test_file_progress_large_numbers_and_no_files(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            make_game.file_progress("checked", lambda: 0.0)(102185, 102185)
            make_game.file_progress("checked", lambda: 0.0)(0, 0)
        self.assertIn("checked 102,185 of 102,185 files (100%, 0 s)", out.getvalue().splitlines())
        self.assertEqual(out.getvalue().count("@@progress 1.000"), 2)

    def test_the_check_is_announced_and_reports(self):
        source = (ROOT / "nx2/make_game.py").read_text(encoding="utf-8")
        body = source[source.index("def stand_in_damaged_files("):]
        body = body[:body.index("\ndef ")]
        self.assertIn("checking your game files for damaged textures", body)
        self.assertIn('nup.stand_in_damaged(romfs, file_progress("checked"))', body)


if __name__ == "__main__":
    unittest.main()
