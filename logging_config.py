#!/usr/bin/env python3
"""
Secure Logging Configuration Module

This module provides secure logging configuration with:
- Structured logging using Python logging module
- Sensitive data sanitization
- Log rotation and size limits
- Separate log files for app and worker processes
"""

import os
import re
import logging
import logging.handlers
from typing import Optional, List, Any
from config import LoggingConfig


class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data in log messages."""

    def __init__(self, patterns: List[str]):
        super().__init__()
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._sanitize_message(record.msg)
        if hasattr(record, 'message'):
            record.message = self._sanitize_message(record.message)
        return True

    def _sanitize_message(self, message: str) -> str:
        """Sanitize sensitive data from log message."""
        sanitized = message
        for pattern in self.patterns:
            sanitized = pattern.sub('[REDACTED]', sanitized)
        return sanitized


class SecurityEventLogger:
    """Specialized logger for security events and metrics."""

    def __init__(self, config: LoggingConfig):
        """Initialize security event logger."""
        self.config = config
        self.security_logger = None
        self.metrics_logger = None
        self._setup_security_logging()

    def _setup_security_logging(self):
        """Set up dedicated security and metrics logging."""
        # Security events logger
        self.security_logger = logging.getLogger('security_events')
        self.security_logger.setLevel(logging.INFO)
        self.security_logger.propagate = False  # Don't propagate to root logger

        # Metrics logger
        self.metrics_logger = logging.getLogger('security_metrics')
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.propagate = False

        # Create formatters
        security_formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(event_type)s - %(message)s',
            datefmt=self.config.date_format
        )

        metrics_formatter = logging.Formatter(
            '%(asctime)s - METRICS - %(metric_type)s - %(metric_name)s - %(value)s - %(message)s',
            datefmt=self.config.date_format
        )

        # Create handlers for both loggers
        for logger_name, logger, formatter, filename in [
            ('security', self.security_logger, security_formatter, 'logs/security_events.log'),
            ('metrics', self.metrics_logger, metrics_formatter, 'logs/security_metrics.log')
        ]:
            # Ensure log directory exists
            log_dir = os.path.dirname(filename)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            # Create rotating file handler
            handler = logging.handlers.RotatingFileHandler(
                filename,
                maxBytes=self.config.max_log_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def log_security_event(self, event_type: str, message: str, **kwargs):
        """Log a security event."""
        if self.security_logger:
            # Create a log record with additional fields
            extra = {
                'event_type': event_type,
                **kwargs
            }
            self.security_logger.info(message, extra=extra)

    def log_security_metric(self, metric_type: str, metric_name: str, value: Any, message: str = "", **kwargs):
        """Log a security metric."""
        if self.metrics_logger:
            extra = {
                'metric_type': metric_type,
                'metric_name': metric_name,
                'value': value,
                **kwargs
            }
            self.metrics_logger.info(message, extra=extra)

    def log_file_upload_attempt(self, filename: str, file_size: int, success: bool, user_id: Optional[str] = None, **kwargs):
        """Log file upload attempt."""
        event_type = 'FILE_UPLOAD_SUCCESS' if success else 'FILE_UPLOAD_FAILED'
        self.log_security_event(
            event_type,
            f"File upload: {filename} ({file_size} bytes)",
            uploaded_filename=filename,
            file_size=file_size,
            user_id=user_id or 'anonymous',
            **kwargs
        )

    def log_validation_failure(self, validation_type: str, input_value: str, reason: str, **kwargs):
        """Log input validation failure."""
        self.log_security_event(
            'VALIDATION_FAILURE',
            f"{validation_type} validation failed: {reason}",
            validation_type=validation_type,
            input_value=input_value[:100] + '...' if len(input_value) > 100 else input_value,
            reason=reason,
            **kwargs
        )

    def log_anomalous_activity(self, activity_type: str, description: str, severity: str = 'medium', **kwargs):
        """Log anomalous activity detection."""
        self.log_security_event(
            'ANOMALOUS_ACTIVITY',
            f"Anomalous {activity_type}: {description}",
            activity_type=activity_type,
            severity=severity,
            **kwargs
        )

    def log_rate_limit_exceeded(self, endpoint: str, client_ip: str, request_count: int, **kwargs):
        """Log rate limit exceeded."""
        self.log_security_event(
            'RATE_LIMIT_EXCEEDED',
            f"Rate limit exceeded for {endpoint} from {client_ip}",
            endpoint=endpoint,
            client_ip=client_ip,
            request_count=request_count,
            **kwargs
        )

    def increment_metric(self, metric_name: str, value: int = 1, **kwargs):
        """Increment a security metric counter."""
        self.log_security_metric(
            'counter',
            metric_name,
            value,
            f"Incremented {metric_name} by {value}",
            **kwargs
        )

    def record_metric(self, metric_name: str, value: Any, metric_type: str = 'gauge', **kwargs):
        """Record a security metric value."""
        self.log_security_metric(
            metric_type,
            metric_name,
            value,
            f"Recorded {metric_name}: {value}",
            **kwargs
        )


# Global security event logger instance
_security_logger = None

def get_security_logger(config: Optional[LoggingConfig] = None) -> SecurityEventLogger:
    """Get the global security event logger instance."""
    global _security_logger
    if _security_logger is None:
        if config is None:
            from config import get_config
            config = get_config().logging
        _security_logger = SecurityEventLogger(config)
    return _security_logger


def setup_logging(config: LoggingConfig, logger_name: str = "app") -> logging.Logger:
    """
    Set up secure logging configuration.

    Args:
        config: Logging configuration
        logger_name: Name for the logger ('app' or 'worker')

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(config.app_log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Determine log file and level based on logger name
    if logger_name == "worker":
        log_file = config.worker_log_file
        log_level = getattr(logging, config.worker_log_level.upper(), logging.INFO)
    else:
        log_file = config.app_log_file
        log_level = getattr(logging, config.app_log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        config.log_format,
        datefmt=config.date_format
    )

    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.max_log_size_mb * 1024 * 1024,  # Convert MB to bytes
        backupCount=config.backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    # Add sensitive data filter if enabled
    if config.mask_sensitive_data:
        sensitive_filter = SensitiveDataFilter(config.sensitive_patterns)
        file_handler.addFilter(sensitive_filter)

    # Add handler to logger
    logger.addHandler(file_handler)

    # Also log to console for development (without sensitive data masking for readability)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        config: Optional logging config (uses default if not provided)

    Returns:
        Logger instance
    """
    if config is None:
        from config import get_config
        config = get_config().logging

    # Check if logger already exists
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    # Set up logger
    if name.startswith("worker"):
        return setup_logging(config, "worker")
    else:
        return setup_logging(config, "app")


# Convenience functions
def get_app_logger() -> logging.Logger:
    """Get the application logger."""
    return get_logger("app")


def get_worker_logger() -> logging.Logger:
    """Get the worker logger."""
    return get_logger("worker")


# Global loggers for backward compatibility
app_logger = None
worker_logger = None


def initialize_loggers(config: Optional[LoggingConfig] = None):
    """Initialize global loggers."""
    global app_logger, worker_logger

    if config is None:
        from config import get_config
        config = get_config().logging

    app_logger = setup_logging(config, "app")
    worker_logger = setup_logging(config, "worker")