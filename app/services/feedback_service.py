from sqlalchemy.orm import Session

from app.core.exceptions import AppError, ErrorCode
from app.core.logging import get_logger
from app.observability.langfuse_client import get_langfuse_client
from app.schemas.feedback import FeedbackData, FeedbackRequest
from app.utils.id_generator import generate_uuid

logger = get_logger(__name__)

FEEDBACK_SCORE_NAME = "user_feedback"


class FeedbackService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def submit_feedback(self, tenant_id: str, request: FeedbackRequest) -> FeedbackData:
        if request.score < 1 or request.score > 5:
            raise AppError(
                ErrorCode.FEEDBACK_INVALID_SCORE,
                context={"score": request.score},
            )

        langfuse = get_langfuse_client()
        if langfuse is None:
            raise AppError(ErrorCode.FEEDBACK_FAILED, msg="Langfuse 未启用或不可用")

        try:
            trace = langfuse.fetch_trace(request.trace_id).data
            metadata = getattr(trace, "metadata", None) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            if metadata.get("tenant_id") != tenant_id:
                raise AppError(
                    ErrorCode.FEEDBACK_LOG_MISMATCH,
                    context={"trace_id": request.trace_id},
                )
            if request.log_id and metadata.get("log_id") != request.log_id:
                raise AppError(
                    ErrorCode.FEEDBACK_LOG_MISMATCH,
                    context={
                        "log_id": request.log_id,
                        "trace_id": request.trace_id,
                    },
                )
        except AppError:
            raise
        except Exception as exc:
            logger.warning(
                "FEEDBACK_TRACE_VALIDATION_FAILED | trace_id=%s | error_type=%s",
                request.trace_id,
                type(exc).__name__,
            )
            raise AppError(
                ErrorCode.FEEDBACK_LOG_MISMATCH,
                context={"trace_id": request.trace_id},
            ) from exc

        feedback_id = generate_uuid()
        try:
            langfuse.score(
                name=FEEDBACK_SCORE_NAME,
                value=request.score,
                trace_id=request.trace_id,
                id=feedback_id,
                comment=request.comment,
            )
            langfuse.flush()
        except Exception as exc:
            logger.warning(
                "FEEDBACK_SUBMIT_FAILED | trace_id=%s | log_id=%s | error_type=%s | error=%s",
                request.trace_id,
                request.log_id,
                type(exc).__name__,
                str(exc),
            )
            raise AppError(
                ErrorCode.FEEDBACK_FAILED,
                internal_msg=str(exc),
                context={"trace_id": request.trace_id},
            ) from exc

        return FeedbackData(
            feedback_id=feedback_id,
            trace_id=request.trace_id,
            log_id=request.log_id,
            score=request.score,
        )
