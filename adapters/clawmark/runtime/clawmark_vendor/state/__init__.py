"""Register only the filesystem/email managers included in this release."""
from .base import BaseStateManager
from . import filesystem, email

__all__ = ['BaseStateManager']
