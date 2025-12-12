# app_logger.py
import os
import logging
import json
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


# Configuration - moved to class for better organization
class LoggerConfig:
    """Logger configuration constants."""

    BACKUP_COUNT = 30  # Keep logs for 30 days
    WHEN = "midnight"  # Rotate at midnight
    INTERVAL = 1  # Daily rotation
    ENCODING = "utf-8"
    FORMAT = "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LOG_FILE = "app.log"


def setup_logger(
    name: str = "api.application", log_level: int = logging.INFO, enable_console: bool = True
) -> logging.Logger:
    """Configure and return a logger instance with daily log rotation."""

    # Disable uvicorn access logs to reduce noise
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

    # Create logger
    logger = logging.getLogger(name)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Don't propagate to root logger
    logger.propagate = False

    # Only set level if it's not already set
    if logger.level == logging.NOTSET:
        logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(fmt=LoggerConfig.FORMAT, datefmt=LoggerConfig.DATE_FORMAT)

    # Console handler (if enabled)
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)

    # File handler with daily rotation
    log_filepath = os.path.join(LOG_DIR, LoggerConfig.LOG_FILE)
    file_handler = TimedRotatingFileHandler(
        filename=log_filepath,
        when=LoggerConfig.WHEN,
        interval=LoggerConfig.INTERVAL,
        backupCount=LoggerConfig.BACKUP_COUNT,
        encoding=LoggerConfig.ENCODING,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)

    # Log startup message
    logger.info(f"Logger initialized. Logs stored in: {LOG_DIR}")
    logger.info(f"Rotation: daily at midnight, keeping {LoggerConfig.BACKUP_COUNT} days of backups")

    return logger


def get_rotation_info() -> Dict[str, Any]:
    """Get information about log rotation settings."""
    log_filepath = os.path.join(LOG_DIR, LoggerConfig.LOG_FILE)

    # Check existing backup files
    backup_files = []
    if os.path.exists(LOG_DIR):
        for f in os.listdir(LOG_DIR):
            if f.startswith(f"{LoggerConfig.LOG_FILE}."):
                backup_files.append(f)

    # Get current log file size
    current_log_size = 0
    if os.path.exists(log_filepath):
        current_log_size = os.path.getsize(log_filepath)

    return {
        "log_dir": LOG_DIR,
        "current_log": LoggerConfig.LOG_FILE,
        "current_log_size_mb": round(current_log_size / (1024 * 1024), 2),
        "backup_count": LoggerConfig.BACKUP_COUNT,
        "backup_count_actual": len(backup_files),
        "rotation_interval": f"{LoggerConfig.INTERVAL} day(s)",
        "rotation_time": LoggerConfig.WHEN,
        "existing_backups": sorted(backup_files),
        "last_modified": (
            datetime.fromtimestamp(os.path.getmtime(log_filepath)).isoformat() if os.path.exists(log_filepath) else None
        ),
    }


def rotate_logs_now() -> bool:
    """Manually trigger log rotation."""
    for handler in app_logger.handlers:
        if isinstance(handler, TimedRotatingFileHandler):
            try:
                handler.doRollover()
                app_logger.info("Manual log rotation completed")
                return True
            except Exception as e:
                app_logger.error(f"Failed to rotate logs: {e}")
                return False
    app_logger.warning("No TimedRotatingFileHandler found")
    return False


def cleanup_old_logs(days_to_keep: Optional[int] = None) -> List[str]:
    """Clean up log files older than specified days."""
    if days_to_keep is None:
        days_to_keep = LoggerConfig.BACKUP_COUNT

    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    deleted_files = []

    for filename in os.listdir(LOG_DIR):
        if filename.startswith(f"{LoggerConfig.LOG_FILE}."):
            try:
                # Extract date from filename (app.log.2024-01-01)
                date_str = filename.split(f"{LoggerConfig.LOG_FILE}.")[1]
                # Handle different date formats
                for fmt in ["%Y-%m-%d", "%Y%m%d"]:
                    try:
                        file_date = datetime.strptime(date_str, fmt)
                        if file_date < cutoff_date:
                            filepath = os.path.join(LOG_DIR, filename)
                            os.remove(filepath)
                            deleted_files.append(filename)
                            break
                    except ValueError:
                        continue
            except Exception as e:
                # Use root logger since we can't guarantee app_logger exists
                logging.getLogger().warning(f"Could not parse date from {filename}: {e}")

    return deleted_files


def get_log_tail(lines: int = 100) -> List[str]:
    """Get the last N lines from the current log file."""
    log_filepath = os.path.join(LOG_DIR, LoggerConfig.LOG_FILE)
    if not os.path.exists(log_filepath):
        return []

    try:
        with open(log_filepath, "r", encoding=LoggerConfig.ENCODING) as f:
            return f.readlines()[-lines:]
    except Exception:
        return []


# Global logger instance - YES, export this
app_logger = setup_logger()

if __name__ == "__main__":
    # Test the logger
    app_logger.debug("Debug message (should not appear)")
    app_logger.info("Info message")
    app_logger.warning("Warning message")
    app_logger.error("Error message")

    # Show rotation info
    info = get_rotation_info()
    print("\nLog Rotation Information:")
    print(json.dumps(info, indent=2))
