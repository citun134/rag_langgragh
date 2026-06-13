import uuid
from qdrant_client.http import models as qmodels


ROLE_ALLOWED_VISIBILITIES = {
    "employee": ["public"],
    "manager": ["public", "internal"],
    "hr": ["public", "internal", "confidential"],
    "admin": ["public", "internal", "confidential", "private"],
}


VALID_VISIBILITIES = {
    "public",
    "internal",
    "confidential",
    "private",
}


# LangChain QdrantVectorStore thường lưu Document.metadata dưới payload key "metadata".
# Nếu filter không ra kết quả, đổi thành "" rồi test lại.
METADATA_PREFIX = "metadata."


def meta_key(name: str) -> str:
    return f"{METADATA_PREFIX}{name}"


def normalize_role(role: str) -> str:
    role = str(role or "employee").lower().strip()

    if role not in ROLE_ALLOWED_VISIBILITIES:
        return "employee"

    return role


def normalize_visibility(visibility: str) -> str:
    visibility = str(visibility or "private").lower().strip()

    if visibility not in VALID_VISIBILITIES:
        raise ValueError(
            "visibility must be one of: public, internal, confidential, private"
        )

    return visibility


def build_document_metadata(
    visibility: str,
    owner_user_id: str,
    doc_id: str | None = None,
) -> dict:
    """
    Build metadata for one uploaded document.

    Rules:
    - public/internal/confidential: accessed by role.
    - private: only owner_user_id can access.
    """

    visibility = normalize_visibility(visibility)
    owner_user_id = str(owner_user_id or "").strip()

    if not owner_user_id:
        raise ValueError("owner_user_id is required")

    return {
        "doc_id": doc_id or str(uuid.uuid4()),
        "visibility": visibility,
        "owner_user_id": owner_user_id,
    }


def build_access_filter(
    user_id: str,
    role: str = "employee",
):
    """
    Build Qdrant filter.

    Rules:
    - employee: public + own private
    - manager: public + internal + own private
    - hr: public + internal + confidential + own private
    - admin: everything
    """

    user_id = str(user_id or "anonymous").strip()
    role = normalize_role(role)

    if role == "admin":
        return None

    allowed_visibilities = ROLE_ALLOWED_VISIBILITIES[role]

    return qmodels.Filter(
        should=[
            # Role-based documents: public / internal / confidential
            qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key=meta_key("visibility"),
                        match=qmodels.MatchAny(any=allowed_visibilities),
                    )
                ]
            ),

            # Private document: only owner can see it
            qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key=meta_key("visibility"),
                        match=qmodels.MatchValue(value="private"),
                    ),
                    qmodels.FieldCondition(
                        key=meta_key("owner_user_id"),
                        match=qmodels.MatchValue(value=user_id),
                    ),
                ]
            ),
        ]
    )