"""Generic optional criterion observation, with state local to one invocation.

Imports of the analyzer are deferred until a plugin is actually attached. No
observation result is put in the selection handler or its decision log.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from inspect import Parameter, signature


INHERIT_INFORMATION = object()
_controller = ContextVar("information_observation_controller", default=None)
_active = ContextVar("information_observation_active_modules", default=())


@dataclass(frozen=True)
class StageRemoval:
    stage_id: str
    module_name: str
    module_type: str
    discarded_segment_ids: tuple[str, ...]
    explicit_discard_action_ids: tuple[str, ...]
    outcome_removed_segment_ids: tuple[str, ...]
    available_segment_ids: tuple[str, ...]
    current_discarded_segment_ids: tuple[str, ...]


def module_names(module):
    cls = type(module)
    return (module.name, cls.__name__, f"{cls.__module__}.{cls.__qualname__}")


class InformationObservation:
    """A pipeline's baseline and effective stage removals; never reused across runs."""

    def __init__(self, analyzer, report, *, on_report=None):
        from open_video_summary.core.summarizers.information_plugin import InformationAnalysisPlugin
        self.plugin = analyzer if isinstance(analyzer, InformationAnalysisPlugin) else InformationAnalysisPlugin(analyzer)
        self.initial_report = self.report = report
        self.on_report = on_report
        self.virtual_removed = set()
        self.sequence = 0
        self.last_error = None

    def excluded(self, module, binding):
        if binding is None or binding is False:
            return True
        excluded = set(self.plugin.excluded_modules)
        if binding is not INHERIT_INFORMATION:
            excluded.update(getattr(binding, "excluded_modules", ()))
            excluded.update(getattr(getattr(binding, "config", None), "excluded_modules", ()))
        return bool(excluded.intersection(module_names(module)))

    @staticmethod
    def _items(handler):
        return tuple((f"v{vi}:s{si}", segment)
                     for vi, video in enumerate(handler.source)
                     for si, segment in enumerate(video.segments))

    def state(self, handler):
        items = self._items(handler)
        restored = {identifier for identifier, segment in items
                    if segment in handler.output or segment in handler.include}
        self.virtual_removed.difference_update(restored)
        actual = {identifier for identifier, segment in items
                  if segment in handler.discard and segment not in handler.output}
        return actual | self.virtual_removed

    def evaluate(self, module, handler, invoke):
        self.sequence += 1
        stage_index = self.sequence
        binding = getattr(module, "information_analyzer", INHERIT_INFORMATION)
        before = self.state(handler)
        log_lengths = {name: len(log.discard) for name, log in handler.agent_logs.items()}
        # The original return value and exceptions remain the criterion's own.
        result = invoke()
        items = self._items(result)
        log = result.agent_logs.get(module.name)
        actions = tuple(log.discard[log_lengths.get(module.name, 0):]) if log else ()
        action_ids = tuple(identifier for identifier, segment in items
                           if any(segment is action for action in actions))
        hook = getattr(module, "information_removed_segments", None)
        hook_error = None
        try:
            outcome = tuple(hook(result)) if callable(hook) else ()
        except Exception as exc:
            hook_error = exc
            self.last_error = type(exc).__name__
            outcome = ()
        outcome_ids = {identifier for identifier, segment in items
                       if any(segment == item for item in outcome)
                       and segment not in result.output and segment not in result.include}
        self.virtual_removed.update(outcome_ids)
        after = self.state(result)
        new = after - before
        # For logged decisions, attribute only this module's effective actions.
        new_actions = any(len(value.discard) > log_lengths.get(name, 0)
                          for name, value in result.agent_logs.items())
        if new_actions:
            attributable = {identifier for identifier, segment in items
                            if any(segment == action for action in actions)} | outcome_ids
            new.intersection_update(attributable)
        if not new or self.excluded(module, binding):
            return result
        order = self.initial_report.snapshot.source_order
        stage = StageRemoval(
            f"{stage_index}:{module.name}", module.name, module_names(module)[2],
            tuple(identifier for identifier in order if identifier in new),
            action_ids, tuple(identifier for identifier in order if identifier in new & outcome_ids),
            tuple(identifier for identifier in order if identifier not in after),
            tuple(identifier for identifier in order if identifier in after),
        )
        from dataclasses import replace
        from open_video_summary.core.summarizers.information_plugin import (
            failed_stage_report, InformationAnalysisPlugin,
        )
        analyzer = self.plugin if binding is INHERIT_INFORMATION else binding
        try:
            if hook_error is not None:
                raise hook_error
            plugin = analyzer if isinstance(analyzer, InformationAnalysisPlugin) else InformationAnalysisPlugin(analyzer)
            stage_report = plugin.analyze_discarded(self.initial_report, stage)
        except Exception as exc:
            stage_report = failed_stage_report(self.initial_report, stage, exc)
        previous = self.report
        self.report = replace(self.report, stage_reports=self.report.stage_reports + (stage_report,))
        try:
            module.last_information_report = self.report
        except Exception:
            pass
        if self.on_report is not None:
            try:
                updated = self.on_report(previous, self.report)
                if updated is not None:
                    self.report = updated
            except Exception:
                pass
        try:
            module.last_information_report = self.report
        except Exception:
            pass
        return result


@contextmanager
def information_observation(controller):
    token = _controller.set(controller)
    try:
        yield controller
    finally:
        _controller.reset(token)


def observe_evaluation(module, handler, invoke):
    """Observe existing or future criteria, including direct library calls."""
    if id(module) in _active.get():
        return invoke()
    controller = _controller.get()
    binding = getattr(module, "information_analyzer", INHERIT_INFORMATION)
    try:
        module.last_information_report = None
        module.last_information_error = None
    except Exception:
        pass
    if controller is None and (binding is INHERIT_INFORMATION or binding is None or binding is False):
        return invoke()
    snapshot = None
    try:
        if controller is None:
            from open_video_summary.core.summarizers.information_contracts import capture_snapshot
            from open_video_summary.core.summarizers.information_plugin import InformationAnalysisPlugin
            plugin = binding if isinstance(binding, InformationAnalysisPlugin) else InformationAnalysisPlugin(binding)
            if not set(plugin.excluded_modules).intersection(module_names(module)):
                snapshot = capture_snapshot(handler.source, selection_stage_order=(module.name,))
                report = plugin.analyze(snapshot)
                controller = InformationObservation(plugin, report)
                module.last_information_report = report
    except Exception as exc:
        try:
            module.last_information_error = type(exc).__name__
            if snapshot is not None:
                from open_video_summary.core.summarizers.information_analysis import failed_information_report
                module.last_information_report = failed_information_report(snapshot, exc)
        except Exception:
            pass
        return invoke()
    if controller is None:
        return invoke()
    token = _active.set(_active.get() + (id(module),))
    context_token = _controller.set(controller)
    try:
        # Capture failures in observer setup/postprocessing, but never retry an
        # already executed criterion or mask its own exception.
        invoked = False
        result = None
        criterion_error = None

        def once():
            nonlocal invoked, result, criterion_error
            invoked = True
            try:
                result = invoke()
                return result
            except BaseException as exc:
                criterion_error = exc
                raise

        try:
            return controller.evaluate(module, handler, once)
        except Exception as exc:
            if criterion_error is not None:
                raise
            controller.last_error = type(exc).__name__
            try:
                module.last_information_error = controller.last_error
            except Exception:
                pass
            return result if invoked else invoke()
    finally:
        _controller.reset(context_token)
        _active.reset(token)


def extend_criterion(cls):
    """Add optional injection and a stage observer without changing positional APIs."""
    if "__init__" in cls.__dict__:
        original = cls.__dict__["__init__"]
        original_signature = signature(original)
        declares_analyzer = "information_analyzer" in original_signature.parameters

        @wraps(original)
        def initialize(self, *args, information_analyzer=INHERIT_INFORMATION, **kwargs):
            if declares_analyzer and information_analyzer is not INHERIT_INFORMATION:
                kwargs["information_analyzer"] = information_analyzer
            original(self, *args, **kwargs)
            if declares_analyzer and information_analyzer is INHERIT_INFORMATION:
                information_analyzer = original_signature.bind(self, *args, **kwargs).arguments.get(
                    "information_analyzer", INHERIT_INFORMATION)
            if information_analyzer is not INHERIT_INFORMATION:
                self.information_analyzer = information_analyzer

        parameters = list(original_signature.parameters.values())
        if "information_analyzer" not in {item.name for item in parameters}:
            parameter = Parameter("information_analyzer", Parameter.KEYWORD_ONLY,
                                  default=INHERIT_INFORMATION)
            insertion = next((index for index, item in enumerate(parameters)
                              if item.kind == Parameter.VAR_KEYWORD), len(parameters))
            parameters.insert(insertion, parameter)
            initialize.__signature__ = original_signature.replace(parameters=parameters)
        cls.__init__ = initialize
    if "evaluate" in cls.__dict__:
        original_evaluate = cls.__dict__["evaluate"]
        handler_parameter = tuple(signature(original_evaluate).parameters)[1]

        @wraps(original_evaluate)
        def evaluate(self, *args, **kwargs):
            handler = args[0] if args else kwargs.get(handler_parameter)
            if handler is None:
                return original_evaluate(self, *args, **kwargs)
            return observe_evaluation(self, handler, lambda: original_evaluate(self, *args, **kwargs))

        cls.evaluate = evaluate
