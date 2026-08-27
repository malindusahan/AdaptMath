"""Focused learning-state repository facade."""
from src.database.repositories.supporting_memory_repository import SupportingMemoryRepository
class LearningStateRepository(SupportingMemoryRepository):
    pass
__all__=["LearningStateRepository"]
