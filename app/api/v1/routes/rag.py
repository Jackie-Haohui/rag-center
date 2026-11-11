from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from app.api.v1.deps import get_current_tenant
from app.core.auth import TenantContext
from app.core.http_clients import ExternalHttpClients, get_external_http_clients
from app.db.session import get_db
from app.schemas.common import success_response
from app.schemas.feedback import FeedbackRequest
from app.schemas.rag import RetrieveRequest
from app.schemas.research import ResearchRequest
from app.services.feedback_service import FeedbackService
from app.services.rag_service import RagService
from app.services.research_service import ResearchService

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/retrieve")
async def retrieve(
    request: RetrieveRequest,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant),
    http_clients: ExternalHttpClients = Depends(get_external_http_clients),
) -> dict:
    data = await RagService(db, http_clients=http_clients).retrieve(
        tenant.tenant_id,
        request,
    )
    return success_response(data.model_dump(mode="json"))


@router.post("/research")
async def research(
    request: ResearchRequest,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant),
    http_clients: ExternalHttpClients = Depends(get_external_http_clients),
    x_eval_allow_supplement: bool | None = Header(
        default=None,
        alias="X-Eval-Allow-Supplement",
        include_in_schema=False,
    ),
) -> dict:
    # The hidden switch is consumed only by the offline A/B runner. Normal API
    # clients always execute the complete bounded Research flow.
    service = ResearchService(db, http_clients=http_clients)
    if x_eval_allow_supplement is False:
        data = await service.research(tenant.tenant_id, request, allow_supplement=False)
    else:
        data = await service.research(tenant.tenant_id, request)
    return success_response(data.model_dump(mode="json"))


@router.post("/feedback")
def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
    tenant: TenantContext = Depends(get_current_tenant),
) -> dict:
    data = FeedbackService(db).submit_feedback(tenant.tenant_id, request)
    return success_response(data.model_dump(mode="json"))
