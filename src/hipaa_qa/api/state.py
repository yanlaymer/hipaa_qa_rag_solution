"""Application state management."""

from ..config import Settings, get_settings
from ..database import DatabaseManager, get_database
from ..services import EmbeddingService, IngestionService, QAService


class AppState:
    """Application state container."""
    
    def __init__(self) -> None:
        self.settings: Settings = get_settings()
        self.db_manager: DatabaseManager = get_database()
        self.embedding_service: EmbeddingService = EmbeddingService(self.settings)
        self.ingestion_service: IngestionService = IngestionService(
            self.db_manager, self.embedding_service
        )
        self.qa_service: QAService = QAService(
            self.db_manager, self.embedding_service, self.settings
        )


# Global app state
_app_state: AppState = None


def get_app_state() -> AppState:
    """Get the global application state."""
    global _app_state
    if _app_state is None:
        _app_state = AppState()
    return _app_state