"""Focused misconception repository facade."""
from src.database.repositories.supporting_memory_repository import SupportingMemoryRepository, normalize_error
class MisconceptionRepository(SupportingMemoryRepository):
    pass
__all__=["MisconceptionRepository","normalize_error"]
