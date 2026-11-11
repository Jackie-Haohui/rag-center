from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import AppError, ErrorCode
from app.schemas.feedback import FeedbackRequest
from app.services import feedback_service as module


class FakeLangfuse:
    def __init__(self, *, tenant_id="tenant", log_id="log-1", fetch_error=None):
        self.metadata = {"tenant_id": tenant_id, "log_id": log_id}
        self.fetch_error = fetch_error
        self.scores = []
        self.flushed = False

    def fetch_trace(self, trace_id):
        if self.fetch_error:
            raise self.fetch_error
        return SimpleNamespace(data=SimpleNamespace(metadata=self.metadata))

    def score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        self.flushed = True


def test_feedback_validates_tenant_and_correlation_id_from_langfuse(monkeypatch):
    langfuse = FakeLangfuse()
    monkeypatch.setattr(module, "get_langfuse_client", lambda: langfuse)
    result = module.FeedbackService(MagicMock()).submit_feedback(
        "tenant",
        FeedbackRequest(trace_id="trace-1", log_id="log-1", score=5),
    )
    assert result.trace_id == "trace-1"
    assert langfuse.scores[0]["trace_id"] == "trace-1"
    assert langfuse.flushed is True


@pytest.mark.parametrize(
    "langfuse",
    [
        FakeLangfuse(tenant_id="other"),
        FakeLangfuse(log_id="other"),
        FakeLangfuse(fetch_error=RuntimeError("not found")),
    ],
)
def test_feedback_rejects_trace_outside_current_tenant_or_correlation(langfuse, monkeypatch):
    monkeypatch.setattr(module, "get_langfuse_client", lambda: langfuse)
    with pytest.raises(AppError) as exc_info:
        module.FeedbackService(MagicMock()).submit_feedback(
            "tenant",
            FeedbackRequest(trace_id="trace-1", log_id="log-1", score=5),
        )
    assert exc_info.value.code == ErrorCode.FEEDBACK_LOG_MISMATCH.code
    assert langfuse.scores == []
