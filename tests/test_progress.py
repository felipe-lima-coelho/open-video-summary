"""Offline regressions for optional heartbeat startup and concurrent scopes."""

from _thread import start_new_thread
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread
import gc
import time
import unittest
import weakref
from unittest.mock import Mock, patch

from open_video_summary.core.summarizers.information_analysis import InformationAnalyzer
from open_video_summary.core.summarizers.information_config import InformationAnalysisConfig
from open_video_summary.core.summarizers.information_contracts import capture_snapshot
from open_video_summary.utils import progress
from tests.test_information_analysis import ScriptedGenerator, SyntheticEvaluator, candidate, videos


class HeartbeatTests(unittest.TestCase):
    def monitor(self):
        starts, stopped = [], Event()

        def start(callback, args):
            starts.append(True)

            def run():
                try:
                    callback(*args)
                finally:
                    stopped.set()

            return start_new_thread(run, ())

        monitor = progress._HeartbeatMonitor(start_thread=start)

        def cleanup():
            monitor.close()
            if starts:
                self.assertTrue(stopped.wait(3), "Heartbeat monitor did not stop.")

        self.addCleanup(cleanup)
        return monitor, starts

    def test_analysis_continues_after_startup_error_or_missing_bootstrap(self):
        source = videos(["Backup diário."])
        items = {("v0:s0", "direct"): [candidate("v0:s0", source[0].segments[0].content)]}

        def analyze(observer=None):
            generator = ScriptedGenerator(items)
            analyzer = InformationAnalyzer(generator, SyntheticEvaluator(),
                InformationAnalysisConfig(qa_enabled=False), progress=observer)
            return analyzer.analyze(capture_snapshot(source)), generator

        baseline, _ = analyze()
        for starter in (Mock(side_effect=MemoryError("synthetic startup failure")),
                        Mock(return_value=123)):
            with self.subTest(failure="synchronous" if starter.side_effect else "bootstrap"):
                # A successful native start can still lose its Python bootstrap.
                # The second starter deliberately never invokes that bootstrap.
                monitor = progress._HeartbeatMonitor(start_thread=starter)
                events = []
                with patch.object(progress, "_MONITOR", monitor):
                    report, generator = analyze(events.append)
                monitor.close()
                self.assertTrue(generator.requests)
                self.assertEqual("completed", report.status)
                self.assertEqual(baseline.counts, report.counts)
                self.assertEqual(baseline.units, report.units)
                self.assertEqual(baseline.candidates, report.candidates)
                self.assertTrue(any(event.event == "call_completed" for event in events))
                self.assertEqual(1, starter.call_count)

    def test_concurrent_and_later_scopes_reuse_one_monitor(self):
        monitor, starts = self.monitor()
        entered = [Event() for _ in range(3)]
        observed = [Event() for _ in range(3)]
        release = Event()

        def work(index):
            with progress.heartbeat(lambda event: observed[index].set(), lambda: index, .001):
                entered[index].set()
                release.wait(3)

        with patch.object(progress, "_MONITOR", monitor), ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(work, index) for index in range(3)]
            try:
                self.assertTrue(all(event.wait(3) for event in entered))
                self.assertTrue(all(event.wait(3) for event in observed))
            finally:
                release.set()
            for future in futures:
                future.result(timeout=3)
            # The monitor stays idle between scopes instead of retiring and
            # racing a new registration into an unserviced queue.
            for _ in range(5):
                tick = Event()
                with progress.heartbeat(lambda event: tick.set(), lambda: None, .001):
                    self.assertTrue(tick.wait(3))
        self.assertEqual(1, len(starts))

    def test_last_scope_can_close_and_reopen_during_an_inflight_tick(self):
        monitor, starts = self.monitor()
        building, release, next_tick = Event(), Event(), Event()
        stale_observer = Mock()

        def build():
            building.set()
            release.wait(3)
            return "closed scope"

        with patch.object(progress, "_MONITOR", monitor):
            try:
                with progress.heartbeat(stale_observer, build, .001):
                    self.assertTrue(building.wait(3))
                with progress.heartbeat(lambda event: next_tick.set(), lambda: "new scope", .001):
                    release.set()
                    self.assertTrue(next_tick.wait(3))
            finally:
                release.set()
        stale_observer.assert_not_called()
        self.assertEqual(1, len(starts))

    def test_slow_observer_cannot_block_provider_or_scope_cleanup(self):
        monitor, _ = self.monitor()
        observing, release, completed = Event(), Event(), Event()
        provider = Mock()
        failures = []

        def slow_observer(event):
            observing.set()
            release.wait(5)

        def work():
            try:
                with progress.heartbeat(slow_observer, lambda: None, .001):
                    if not observing.wait(3):
                        raise AssertionError("The slow observer did not start.")
                    provider()
                with progress.heartbeat(lambda event: None, lambda: None, .001):
                    provider()
            except BaseException as exc:
                failures.append(exc)
            finally:
                completed.set()

        with patch.object(progress, "_MONITOR", monitor):
            worker = Thread(target=work, daemon=True)
            worker.start()
            try:
                finished_before_release = completed.wait(3)
            finally:
                release.set()
                worker.join(timeout=3)
        self.assertTrue(finished_before_release, "Progress blocked the provider workflow.")
        self.assertFalse(worker.is_alive())
        self.assertEqual([], failures)
        self.assertEqual(2, provider.call_count)

    def test_failed_factory_and_observer_do_not_stop_other_heartbeats(self):
        monitor, _ = self.monitor()
        factory_called, observer_called, healthy = Event(), Event(), Event()

        def failed_factory():
            factory_called.set()
            raise MemoryError("synthetic observation failure")

        def failed_observer(event):
            observer_called.set()
            raise RuntimeError("synthetic observer failure")

        with patch.object(progress, "_MONITOR", monitor), \
                progress.heartbeat(lambda event: None, failed_factory, .001), \
                progress.heartbeat(failed_observer, lambda: None, .001), \
                progress.heartbeat(lambda event: healthy.set(), lambda: None, .001):
            self.assertTrue(factory_called.wait(3))
            self.assertTrue(observer_called.wait(3))
            self.assertTrue(healthy.wait(3))

    def test_idle_monitor_releases_finished_scope_closures(self):
        monitor, _ = self.monitor()
        tick = Event()

        class Payload:
            pass

        payload = Payload()
        reference = weakref.ref(payload)
        factory = lambda payload=payload: payload
        with patch.object(progress, "_MONITOR", monitor):
            with progress.heartbeat(lambda event: tick.set(), factory, .001):
                self.assertTrue(tick.wait(3))
        del factory, payload
        deadline = time.monotonic() + 3
        while reference() is not None and time.monotonic() < deadline:
            gc.collect()
            time.sleep(.01)
        self.assertIsNone(reference(), "The idle monitor retained a completed analysis scope.")

    def test_quiet_scope_does_not_start_a_monitor(self):
        starter = Mock()
        monitor = progress._HeartbeatMonitor(start_thread=starter)
        provider = Mock()
        with patch.object(progress, "_MONITOR", monitor):
            with progress.heartbeat(None, lambda: None, .001):
                provider()
        provider.assert_called_once_with()
        starter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
