"""Role-based model routing for ModuLLe.

Lets an application assign different models to different tasks
(main chat, session titles, context compression, vision analysis)
while defaulting sensibly to the main model.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from modulle.utils.logging_config import get_logger

logger = get_logger(__name__)

TASK_ROLES = ('main', 'title', 'compress', 'vision')


@dataclass
class ModelRole:
    """Config for one task role."""
    provider: Optional[str] = None        # None = use main model's provider
    model: Optional[str] = None           # None = use main model
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@dataclass
class TaskRouter:
    """Maps task roles to model overrides.

    Example:
        >>> router = TaskRouter()
        >>> router.set('title', model='llama3.2:1b')
        >>> router.resolve('title')          # -> ModelRole(model='llama3.2:1b')
        >>> router.effective('compress', main_provider='ollama',
        ...                  main_model='llama3')
    """

    roles: Dict[str, ModelRole] = field(default_factory=dict)

    def set(self, role: str, provider: Optional[str] = None,
            model: Optional[str] = None, api_key: Optional[str] = None,
            base_url: Optional[str] = None):
        """Configure a role (creates it if missing)."""
        if role not in TASK_ROLES:
            raise ValueError(f"Unknown role: {role}. Valid: {TASK_ROLES}")
        self.roles[role] = ModelRole(
            provider=provider, model=model,
            api_key=api_key, base_url=base_url)

    def clear(self, role: str):
        """Remove a role's override (falls back to main)."""
        self.roles.pop(role, None)

    def resolve(self, role: str) -> ModelRole:
        """Return the configured ModelRole for a role (or empty default)."""
        return self.roles.get(role, ModelRole())

    def effective(self, role: str, main_provider: str, main_model: str,
                  main_api_key: Optional[str] = None,
                  main_base_url: Optional[str] = None) -> ModelRole:
        """Resolve a role to its effective provider/model, defaulting main.

        'vision' defaults to the main model too — callers should fall back
        to main-model vision capability when no explicit vision override is
        set.
        """
        cfg = self.roles.get(role, ModelRole())
        return ModelRole(
            provider=cfg.provider or main_provider,
            model=cfg.model or main_model,
            api_key=cfg.api_key if cfg.api_key is not None else main_api_key,
            base_url=cfg.base_url if cfg.base_url is not None else main_base_url,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for config storage."""
        return {role: vars(cfg) for role, cfg in self.roles.items()}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "TaskRouter":
        """Build from config data (ignores unknown roles)."""
        router = cls()
        for role, cfg in (data or {}).items():
            if role in TASK_ROLES and isinstance(cfg, dict):
                router.roles[role] = ModelRole(
                    provider=cfg.get('provider'),
                    model=cfg.get('model'),
                    api_key=cfg.get('api_key'),
                    base_url=cfg.get('base_url'),
                )
        return router