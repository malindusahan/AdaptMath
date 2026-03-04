"""Authenticated learner-profile endpoint."""

from fastapi import APIRouter, Depends

from app.clients.memory.memory_client import MemoryAuthenticatedUser
from app.core.user_auth import require_authenticated_student
from app.schemas.student_profile import StudentProfileResponse
from app.services.student_profile_service import get_student_profile_reader

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("", response_model=StudentProfileResponse)
def get_student_profile(
    user: MemoryAuthenticatedUser = Depends(require_authenticated_student),
) -> StudentProfileResponse:
    """Return only the profile belonging to the authenticated student."""
    return get_student_profile_reader().read(
        student_id=user.student_id,
        username=user.username,
        age=user.age,
    )
