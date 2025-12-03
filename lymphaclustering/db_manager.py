# db_manager.py
import numpy as np
import pandas as pd
import threading
import time
import atexit
import logging
import os
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import lancedb
from lancedb.pydantic import LanceModel, Vector

from app_logger import app_logger

# Configuration constants
VECTOR_DIM = 2048
BUFFER_FLUSH_INTERVAL_SECONDS = 6 * 3600
BUFFER_BATCH_LIMIT = 2000


class LymphNodeSchema(LanceModel):
    """Schema for lymph node embeddings in LanceDB."""

    slide_id: str
    wsi_id: str
    vector: Vector(VECTOR_DIM)
    cluster_id: int = -1
    threshold: Optional[float]
    timestamp: datetime
    path: str


class WriteBuffer:
    """
    Accumulates writes in memory and flushes them to LanceDB in bulk.
    This solves the "Write Amplification" bottleneck.
    """

    def __init__(self, db_manager, flush_interval=BUFFER_FLUSH_INTERVAL_SECONDS, batch_limit=BUFFER_BATCH_LIMIT):
        self.buffer = []
        self.lock = threading.Lock()
        self.db_manager = db_manager
        self.flush_interval = flush_interval
        self.batch_limit = batch_limit
        self._stop_event = threading.Event()

        # Start background flusher
        self.thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self.thread.start()

        # Ensure flush on server shutdown
        atexit.register(self.flush)

    def add(self, records: List[Dict]):
        """Add records to buffer. Trigger flush if limit reached."""
        if not records:
            return

        with self.lock:
            self.buffer.extend(records)
            current_len = len(self.buffer)

        if current_len >= self.batch_limit:
            app_logger.info(f"Buffer full ({current_len} items). Triggering flush.")
            self.flush()

    def flush(self):
        """Force write to disk in a separate thread context."""
        if self.buffer and len(self.buffer) > 0:
            app_logger.info(f"Flushing {len(self.buffer)} records into VectorDB")

        to_write = []
        with self.lock:
            if not self.buffer:
                return
            to_write = self.buffer
            self.buffer = []  # Clear buffer immediately

        # Write to DB outside the lock to avoid blocking API reads
        try:
            df = pd.DataFrame(to_write)
            self.db_manager.persist_batch(df)
        except Exception as e:
            app_logger.error(f"Failed to flush buffer: {e}")

    def _periodic_flush(self):
        """Background loop for periodic flushing."""
        while not self._stop_event.is_set():
            time.sleep(self.flush_interval)
            self.flush()

    def get_buffered_vectors(self, slide_ids: List[str]) -> pd.DataFrame:
        """
        Check buffer for vectors that haven't hit disk yet (Read-Your-Writes).
        """
        with self.lock:
            if not self.buffer:
                return pd.DataFrame()
            df = pd.DataFrame(self.buffer)

        if df.empty or "slide_id" not in df.columns:
            return pd.DataFrame()

        matches = df[df["slide_id"].isin(slide_ids)]
        return matches

    def stop(self):
        """Stop the background flush thread."""
        self._stop_event.set()
        self.thread.join()

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            return {
                "buffer_size": len(self.buffer),
                "flush_interval": self.flush_interval,
                "batch_limit": self.batch_limit,
                "thread_alive": self.thread.is_alive(),
            }


class VectorDBManager:
    """Manages vector database operations with write-behind buffering."""

    def __init__(
        self,
        db_path: str,
        table_name: str,
        flush_interval: int = BUFFER_FLUSH_INTERVAL_SECONDS,
        batch_limit: int = BUFFER_BATCH_LIMIT,
    ):
        self.db_path = db_path
        self.table_name = table_name
        self.db = lancedb.connect(self.db_path)
        self.tbl = self._init_table()

        # Initialize the Write Buffer
        self.buffer = WriteBuffer(self, flush_interval, batch_limit)

    def _init_table(self):
        """Initialize or open the LanceDB table."""
        if self.table_name not in self.db.table_names():
            app_logger.info(f"Creating new Vector Table: {self.table_name}")
            return self.db.create_table(self.table_name, schema=LymphNodeSchema)
        else:
            return self.db.open_table(self.table_name)

    def get_existing_vectors(self, slide_ids: List[str]) -> pd.DataFrame:
        """
        Hybrid Cache Lookup: Checks both Memory Buffer and Disk.
        """
        if not slide_ids:
            return pd.DataFrame()

        # 1. Check Disk (LanceDB)
        try:
            quoted = [f"'{s}'" for s in slide_ids]
            if len(quoted) > 5000:
                app_logger.debug("Large batch lookup, scanning full table...")
                disk_df = self.tbl.to_pandas()
                disk_df = disk_df[disk_df["slide_id"].isin(slide_ids)]
            else:
                filter_str = f"slide_id IN ({','.join(quoted)})"
                disk_df = self.tbl.search().where(filter_str).to_pandas()
        except Exception as e:
            app_logger.warning(f"Disk lookup warning: {e}")
            disk_df = pd.DataFrame()

        # 2. Check Memory (WriteBuffer)
        buffer_df = self.buffer.get_buffered_vectors(slide_ids)

        # 3. Merge (Prefer Buffer over Disk if duplicate exists)
        if disk_df.empty and buffer_df.empty:
            return pd.DataFrame()

        combined = pd.concat([disk_df, buffer_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["slide_id"], keep="last")

        return combined

    def persist_batch(self, df: pd.DataFrame):
        """
        Physical write to disk. Called ONLY by WriteBuffer or Maintenance tasks.
        Uses Delete-Insert to reliably handle upserts and avoid Merge schema conflicts.
        """
        if df.empty:
            return
        app_logger.info(f"Persisting {len(df)} records to Disk...")

        # Ensure correct timestamp format
        if "timestamp" in df.columns and df["timestamp"].dtype == "O":
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        try:
            # 1. Extract unique IDs from the incoming batch
            ids_to_update = df["slide_id"].unique()

            # 2. Delete existing records with these IDs (Upsert Step 1)
            if len(ids_to_update) > 0:
                ids_str = ", ".join([f"'{str(sid)}'" for sid in ids_to_update])
                self.tbl.delete(f"slide_id IN ({ids_str})")

            # 3. Add the new/updated records (Upsert Step 2)
            self.tbl.add(df)

        except Exception as e:
            app_logger.error(f"Persist batch failed: {e}")
            # Fallback: Attempt simple add if delete fails
            try:
                self.tbl.add(df)
            except Exception as e2:
                app_logger.error(f"Critical write failure: {e2}")

    def search_similarity(self, query_vector: np.array, limit: int = 5, exclude_id: str = None):
        """Search for similar vectors in the database."""
        search_builder = self.tbl.search(query_vector).limit(limit)
        if exclude_id:
            search_builder = search_builder.where(f"slide_id != '{exclude_id}'")
        return search_builder.to_pandas()

    def _flush_buffer(self) -> Tuple[bool, str]:
        """Flush the write buffer to disk."""
        try:
            buffer_size = len(self.buffer.buffer)
            self.buffer.flush()
            return True, f"Flushed {buffer_size} records from buffer"
        except Exception as e:
            return False, f"Failed to flush buffer: {e}"

    def _perform_compaction(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Perform database compaction and cleanup."""
        try:
            start_time = time.time()

            # Get pre-compaction stats
            pre_stats = self._get_detailed_table_stats()

            # Perform compaction
            compaction_result = self.tbl.compact_files()

            # Cleanup old versions
            cleanup_result = self.tbl.cleanup_old_versions()

            # Try optimization if available
            optimization_result = None
            if hasattr(self.tbl, "optimize"):
                self.tbl.optimize()
                optimization_result = "Table optimized"
            elif hasattr(self.tbl, "vacuum"):
                self.tbl.vacuum()
                optimization_result = "Table vacuumed"

            # Get post-compaction stats
            post_stats = self._get_detailed_table_stats()

            compaction_time = time.time() - start_time

            details = {
                "time_seconds": round(compaction_time, 2),
                "compaction_result": str(compaction_result),
                "cleanup_result": str(cleanup_result),
                "optimization_result": optimization_result,
                "pre_stats": pre_stats,
                "post_stats": post_stats,
            }

            return True, f"Compaction completed in {compaction_time:.2f}s", details

        except Exception as e:
            return False, f"Compaction failed: {e}", {}

    def _check_and_rebuild_indexes(self, force: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
        """Check and optionally rebuild indexes."""
        try:
            index_info = {"indexes_found": 0, "index_names": [], "rebuilt_indexes": []}

            if hasattr(self.tbl, "list_indexes"):
                indexes = self.tbl.list_indexes()
                index_info["indexes_found"] = len(indexes)
                index_info["index_names"] = indexes

                if indexes and force:
                    for idx_name in indexes:
                        try:
                            if hasattr(self.tbl, "recreate_index"):
                                self.tbl.recreate_index(idx_name)
                                index_info["rebuilt_indexes"].append(idx_name)
                        except Exception as idx_e:
                            app_logger.warning(f"Failed to rebuild index {idx_name}: {idx_e}")

            message = f"Found {index_info['indexes_found']} indexes"
            if index_info["rebuilt_indexes"]:
                message += f", rebuilt {len(index_info['rebuilt_indexes'])} indexes"

            return True, message, index_info

        except Exception as e:
            return False, f"Index check failed: {e}", {}

    def _validate_schema(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate database schema."""
        try:
            schema = self.tbl.schema
            schema_info = {
                "field_count": len(schema),
                "field_names": [field.name for field in schema],
                "field_types": {field.name: str(field.type) for field in schema},
            }
            return True, f"Schema validated with {len(schema)} fields", schema_info
        except Exception as e:
            return False, f"Schema validation failed: {e}", {}

    def _cleanup_temp_files(self) -> Tuple[bool, str, Dict[str, Any]]:
        """Clean up temporary files in database directory."""
        try:
            temp_extensions = [".tmp", ".temp", ".lock", ".wal", ".journal"]
            deleted_files = []

            for file_pattern in temp_extensions:
                temp_files = glob.glob(os.path.join(self.db_path, f"*{file_pattern}"))
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                        deleted_files.append(os.path.basename(temp_file))
                    except Exception as e:
                        app_logger.warning(f"Failed to delete {temp_file}: {e}")

            cleanup_info = {"files_deleted": len(deleted_files), "deleted_files": deleted_files}

            return True, f"Cleaned up {len(deleted_files)} temporary files", cleanup_info

        except Exception as e:
            return False, f"Temp cleanup failed: {e}", {}

    def _get_detailed_table_stats(self) -> Dict[str, Any]:
        """Get detailed table statistics including file information."""
        try:
            # Basic stats from table
            df = self.tbl.to_pandas()
            basic_stats = {
                "total_records": len(df),
                "unique_wsi_ids": df["wsi_id"].nunique() if not df.empty else 0,
                "clustered_records": len(df[df["cluster_id"] != -1]) if not df.empty else 0,
                "timestamp_range": (
                    {
                        "min": (
                            df["timestamp"].min().isoformat() if not df.empty and "timestamp" in df.columns else None
                        ),
                        "max": (
                            df["timestamp"].max().isoformat() if not df.empty and "timestamp" in df.columns else None
                        ),
                    }
                    if not df.empty
                    else {}
                ),
            }

            # File system stats
            file_stats = self._get_filesystem_stats()

            return {**basic_stats, **file_stats}

        except Exception as e:
            app_logger.error(f"Error getting detailed stats: {e}")
            return {"error": str(e)}

    def _get_filesystem_stats(self) -> Dict[str, Any]:
        """Get filesystem statistics for the database."""
        try:
            if not os.path.exists(self.db_path):
                return {"error": "Database path does not exist"}

            total_size = 0
            file_count = 0
            file_types = {}

            for root, dirs, files in os.walk(self.db_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    file_count += 1

                    # Count by file extension
                    ext = os.path.splitext(file)[1].lower()
                    file_types[ext] = file_types.get(ext, 0) + 1

            return {
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": file_count,
                "file_types": file_types,
            }
        except Exception as e:
            app_logger.error(f"Error getting filesystem stats: {e}")
            return {"error": str(e)}

    def run_maintenance(self, force: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive database maintenance tasks.

        Args:
            force: If True, perform more aggressive maintenance (e.g., index rebuilding)

        Returns:
            Dictionary with maintenance results
        """
        app_logger.info("=" * 80)
        app_logger.info("🚀 STARTING DATABASE MAINTENANCE")
        app_logger.info("=" * 80)

        maintenance_results = {
            "steps": [],
            "start_time": datetime.now().isoformat(),
            "force_mode": force,
            "database": self.db_path,
            "table": self.table_name,
        }

        def log_step(step_name: str, success: bool, message: str, details: Dict = None):
            """Helper to log and record maintenance steps."""
            step_info = {
                "step": step_name,
                "status": "completed" if success else "failed",
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "details": details or {},
            }
            maintenance_results["steps"].append(step_info)
            status_icon = "✅" if success else "❌"
            app_logger.info(f"{status_icon} {step_name}: {message}")

        # 1. Buffer Flush
        success, message = self._flush_buffer()
        log_step("buffer_flush", success, message)

        # 2. Database Compaction
        success, message, details = self._perform_compaction()
        log_step("compaction", success, message, details)
        maintenance_results["compaction"] = details

        # 3. Index Maintenance
        success, message, details = self._check_and_rebuild_indexes(force)
        log_step("index_maintenance", success, message, details)
        maintenance_results["index_maintenance"] = details

        # 4. Schema Validation
        success, message, details = self._validate_schema()
        log_step("schema_validation", success, message, details)
        maintenance_results["schema_validation"] = details

        # 5. Temporary File Cleanup
        success, message, details = self._cleanup_temp_files()
        log_step("temp_cleanup", success, message, details)
        maintenance_results["temp_cleanup"] = details

        # 6. Final Health Check
        try:
            final_stats = self._get_detailed_table_stats()
            log_step("health_check", True, "Database health check completed", final_stats)
            maintenance_results["final_stats"] = final_stats
        except Exception as e:
            log_step("health_check", False, f"Health check failed: {e}")

        # Final summary
        maintenance_results["end_time"] = datetime.now().isoformat()

        # Calculate total duration
        start_dt = datetime.fromisoformat(maintenance_results["start_time"])
        end_dt = datetime.fromisoformat(maintenance_results["end_time"])
        maintenance_results["total_duration_seconds"] = round((end_dt - start_dt).total_seconds(), 2)

        # Count successful steps
        successful_steps = [step for step in maintenance_results["steps"] if step["status"] == "completed"]
        failed_steps = [step for step in maintenance_results["steps"] if step["status"] == "failed"]

        maintenance_results["summary"] = {
            "total_steps": len(maintenance_results["steps"]),
            "successful_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "overall_status": "success" if len(failed_steps) == 0 else "partial",
        }

        app_logger.info("=" * 80)
        app_logger.info(f"✅ MAINTENANCE COMPLETED in {maintenance_results['total_duration_seconds']}s")
        app_logger.info(
            f"   Steps: {maintenance_results['summary']['successful_steps']}/{maintenance_results['summary']['total_steps']} successful"
        )
        app_logger.info("=" * 80)

        return maintenance_results

    def get_table_stats(self) -> Dict[str, Any]:
        """Get statistics about the table."""
        try:
            df = self.tbl.to_pandas()
            return {
                "total_records": len(df),
                "unique_wsi_ids": df["wsi_id"].nunique() if not df.empty else 0,
                "clustered_records": len(df[df["cluster_id"] != -1]) if not df.empty else 0,
                "columns": list(df.columns) if not df.empty else [],
            }
        except Exception as e:
            app_logger.error(f"Error getting table stats: {e}")
            return {"error": str(e)}

    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            "database": {
                "path": self.db_path,
                "table": self.table_name,
                "table_stats": self.get_table_stats(),
                "filesystem_stats": self._get_filesystem_stats(),
            },
            "buffer": self.buffer.get_stats(),
            "connection_info": {"vector_dim": VECTOR_DIM, "connected": True if self.db else False},
        }

    def close(self):
        """Clean up resources."""
        self.buffer.stop()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
