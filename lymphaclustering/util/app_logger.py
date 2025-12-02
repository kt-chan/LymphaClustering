# app_logger.py
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configuration
LOG_BACKUP_COUNT = 30  # Keep logs for 30 days
LOG_WHEN = "midnight"  # Rotate at midnight
LOG_INTERVAL = 1  # Daily rotation
LOG_ENCODING = "utf-8"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str = "api.application") -> logging.Logger:
    """Configure and return a logger instance with daily log rotation."""

    # Disable uvicorn access logs
    logging.getLogger("uvicorn.access").disabled = True

    # Create logger
    app_logger = logging.getLogger(name)
    app_logger.setLevel(logging.INFO)
    app_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler (stream to stdout)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # File handler with daily rotation
    log_filepath = os.path.join(LOG_DIR, "app.log")
    file_handler = TimedRotatingFileHandler(
        filename=log_filepath, when=LOG_WHEN, interval=LOG_INTERVAL, backupCount=LOG_BACKUP_COUNT, encoding=LOG_ENCODING
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Add handlers
    app_logger.addHandler(console_handler)
    app_logger.addHandler(file_handler)
    app_logger.propagate = False

    # Log startup message
    app_logger.info(f"Logger initialized with daily rotation. Logs stored in: {LOG_DIR}")
    app_logger.info(f"Rotation: daily at midnight, keeping {LOG_BACKUP_COUNT} days of backups")

    return app_logger


def get_rotation_info() -> dict:
    """Get information about log rotation settings."""
    log_filepath = os.path.join(LOG_DIR, "app.log")

    # Check existing backup files
    backup_files = []
    if os.path.exists(log_filepath):
        # Look for rotated files (app.log.2024-01-01, etc.)
        for f in os.listdir(LOG_DIR):
            if f.startswith("app.log.") and f != "app.log":
                backup_files.append(f)

    return {
        "log_dir": LOG_DIR,
        "current_log": "app.log",
        "backup_count": LOG_BACKUP_COUNT,
        "rotation_interval": f"{LOG_INTERVAL} day(s)",
        "rotation_time": LOG_WHEN,
        "existing_backups": sorted(backup_files),
        "backup_count_actual": len(backup_files),
    }


# Global logger instance
app_logger = setup_logger()


# Optional: Function to manually rotate logs
def rotate_logs_now():
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


# Optional: Cleanup function for old logs
def cleanup_old_logs(days_to_keep: int = None):
    """Clean up log files older than specified days."""
    if days_to_keep is None:
        days_to_keep = LOG_BACKUP_COUNT

    from datetime import datetime, timedelta

    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    deleted_files = []

    for filename in os.listdir(LOG_DIR):
        if filename.startswith("app.log."):
            try:
                # Extract date from filename (app.log.2024-01-01)
                date_str = filename.split("app.log.")[1]
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
                app_logger.warning(f"Could not parse date from {filename}: {e}")

    if deleted_files:
        app_logger.info(f"Cleaned up {len(deleted_files)} old log files: {deleted_files}")
    else:
        app_logger.info("No old log files to clean up")

    return deleted_files


if __name__ == "__main__":
    # Test the logger
    app_logger.debug("Debug message (should not appear)")
    app_logger.info("Info message")
    app_logger.warning("Warning message")
    app_logger.error("Error message")

    # Show rotation info
    info = get_rotation_info()
    print("\nLog Rotation Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
