"""File uploads (customer) and downloads (staff only)."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from supportpilot.api import deps
from supportpilot.api.auth import require_staff
from supportpilot.attachments import UploadRejected, save_upload, storage_path
from supportpilot.config import get_settings
from supportpilot.db.models import Attachment
from supportpilot.tickets.schemas import AttachmentOut
from supportpilot.tickets.service import get_chat_session

router = APIRouter(tags=["uploads"])


@router.post("/uploads", response_model=AttachmentOut, status_code=status.HTTP_201_CREATED)
async def upload(
    db: deps.DB,
    file: Annotated[UploadFile, File()],
    session_id: Annotated[str | None, Form()] = None,
):
    """One file per request. Chat uploads pass the session id; contact-form uploads don't."""
    if session_id and get_chat_session(db, session_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Chat session not found")
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)  # never read more than the limit into memory
    try:
        att = save_upload(db, file.filename or "file", data, session_id=session_id)
    except UploadRejected as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from None
    return AttachmentOut.model_validate(att)


@router.get("/admin/attachments/{attachment_id}", dependencies=[Depends(require_staff)])
def download(attachment_id: str, db: deps.DB):
    att = db.get(Attachment, attachment_id)
    path = storage_path(att) if att else None
    if att is None or not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Attachment not found")
    return FileResponse(
        path,
        media_type=att.content_type,
        filename=att.filename,  # Content-Disposition: attachment
        headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"},
    )
